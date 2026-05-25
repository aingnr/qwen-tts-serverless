"""
handler.py — Qwen3 TTS RunPod Serverless Handler
=================================================
공식 API 출처:
  - https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base (HuggingFace 공식 모델카드)
  - https://github.com/QwenLM/Qwen3-TTS (공식 GitHub README + examples/)

확인된 공식 API:
  model.generate_voice_clone(
      text=str,           ← 단일 문자열 or 리스트
      language=str,       ← "auto" 공식 지원 / "Korean" / "English" 등
      ref_audio=str,      ← 로컬 경로 / URL / base64 / (numpy, sr) 모두 허용
      ref_text=str        ← 레퍼런스 오디오 전사 텍스트
  )
  → (wavs_list, sample_rate)

최적화: create_voice_clone_prompt() 캐시
  → 동일 채널(동일 ref_audio URL) 반복 시 음성 특징 추출 생략

n8n TTS_Sub Section → RunPod POST /run
Input  : { mode, text, reference_audio(URL), reference_text, language }
Output : { audio_base64, status }
"""

import os
import io
import base64
import logging
import traceback
import gc

import runpod
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_ID      = os.environ.get("MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
HF_HOME       = os.environ.get("HF_HOME", "/app/models")
REF_AUDIO_DIR = "/app/ref_audio"   # URL 다운로드 실패 시 로컬 폴백 경로

# ── Global: 모델 + voice_clone_prompt 캐시 ────────────────────────────────────
model = None
_prompt_cache: dict = {}   # { ref_audio_url: prompt_items }


# ── 모델 로드 (기동 시 1회) ───────────────────────────────────────────────────

def load_model():
    global model
    log.info(f"[INIT] Qwen3TTSModel 로딩 중: {MODEL_ID}")

    # flash_attention_2 시도 → 실패 시 sdpa 자동 폴백
    for attn in ("flash_attention_2", "sdpa"):
        try:
            model = Qwen3TTSModel.from_pretrained(
                MODEL_ID,
                device_map="cuda:0",
                dtype=torch.bfloat16,
                attn_implementation=attn,
                cache_dir=HF_HOME,
            )
            log.info(f"[INIT] 로딩 완료 (attn={attn})")
            return
        except Exception as e:
            log.warning(f"[INIT] attn={attn} 실패: {e}")

    raise RuntimeError("모델 로딩 실패 (flash_attention_2 / sdpa 모두 실패)")


# ── voice_clone_prompt 캐시 헬퍼 ─────────────────────────────────────────────

def get_voice_clone_prompt(ref_audio_url: str, ref_text: str):
    """
    동일 채널 ref_audio URL → create_voice_clone_prompt 캐시 반환.
    워커 생존 동안 동일 URL 재요청 시 음성 특징 추출 생략 → 처리 속도 향상.

    참고: ref_audio는 URL 직접 전달 가능 (공식 문서 확인)
    """
    if ref_audio_url in _prompt_cache:
        log.info(f"[CACHE] 캐시 히트: {ref_audio_url[:70]}")
        return _prompt_cache[ref_audio_url]

    # 1순위: URL 직접 전달 (공식 지원)
    ref_source = ref_audio_url

    # 2순위: URL 실패 대비 로컬 폴백 경로 확인 (실제 전달은 URL 먼저 시도)
    filename   = ref_audio_url.split("/")[-1].split("?")[0]
    local_path = os.path.join(REF_AUDIO_DIR, filename)

    log.info("[PROMPT] create_voice_clone_prompt 실행 중...")
    try:
        prompt_items = model.create_voice_clone_prompt(
            ref_audio=ref_source,          # URL 직접 전달
            ref_text=ref_text,
            x_vector_only_mode=False,
        )
    except Exception as e:
        log.warning(f"[PROMPT] URL 전달 실패({e}), 로컬 폴백 시도: {local_path}")
        if not os.path.exists(local_path):
            raise RuntimeError(f"ref_audio 취득 실패 — URL: {ref_audio_url}, 로컬: {local_path}")
        prompt_items = model.create_voice_clone_prompt(
            ref_audio=local_path,
            ref_text=ref_text,
            x_vector_only_mode=False,
        )

    _prompt_cache[ref_audio_url] = prompt_items
    log.info("[PROMPT] 완료 (캐시 저장)")
    return prompt_items


# ── RunPod Handler ────────────────────────────────────────────────────────────

def handler(job: dict) -> dict:
    """
    RunPod Serverless 진입점.

    n8n input JSON (TTS_Sub Section → Create TTS w/RunPod node):
    {
        "input": {
            "mode"            : "voice_clone",
            "text"            : "합성할 텍스트 청크",
            "reference_audio" : "https://...채널별_ref.wav",    ← Master Config: TTS_Ref_Audio (AD컬럼)
            "reference_text"  : "레퍼런스 오디오 전사 텍스트",  ← Master Config: TTS_Ref_Text  (AE컬럼)
            "language"        : "auto"   ← "auto" 공식 지원, "Korean"/"English" 등도 가능
        }
    }

    Returns:
    {
        "audio_base64" : "<base64 WAV>",
        "status"       : "success" | "failed"
    }
    """
    job_input = job.get("input", {})
    job_id    = job.get("id", "local")

    # ── Input 파싱 ─────────────────────────────────────────────────────────
    text          = str(job_input.get("text", "")).strip()
    ref_audio_url = str(job_input.get("reference_audio", "")).strip()
    ref_text      = str(job_input.get("reference_text", "")).strip()
    language      = str(job_input.get("language", "auto")).strip()
    # ※ language="auto" → Qwen3-TTS가 텍스트에서 언어 자동 감지 (공식 지원)

    log.info(
        f"[JOB {job_id}] lang={language} | "
        f"text_len={len(text)} | ref={ref_audio_url[:70]}..."
    )

    # ── Validation ─────────────────────────────────────────────────────────
    if not text:
        return {"error": "input.text 가 비어 있습니다.", "status": "failed"}
    if not ref_audio_url:
        return {"error": "input.reference_audio URL 이 비어 있습니다.", "status": "failed"}

    try:
        # ── voice_clone_prompt 취득 (캐시 우선) ───────────────────────────
        prompt_items = get_voice_clone_prompt(ref_audio_url, ref_text)

        # ── 음성 합성 ─────────────────────────────────────────────────────
        # 공식 API: generate_voice_clone(text, language, voice_clone_prompt)
        # ref: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base
        log.info(f"[TTS] 합성 시작 (text_len={len(text)}, lang={language})")
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=prompt_items,
        )
        audio_data = wavs[0].cpu().numpy() if torch.is_tensor(wavs[0]) else wavs[0]
        log.info(f"[TTS] 완료 — sr={sr}, shape={audio_data.shape}")

        # ── 오디오 → base64 WAV ───────────────────────────────────────────
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sr, format="WAV")
        buffer.seek(0)
        audio_b64 = base64.b64encode(buffer.read()).decode("utf-8")

        log.info(f"[JOB {job_id}] 성공 — base64={len(audio_b64)//1024}KB")
        return {
            "audio_base64": audio_b64,
            "status": "success",
        }

    except Exception as e:
        log.error(f"[JOB {job_id}] 에러: {e}\n{traceback.format_exc()}")
        return {
            "error": str(e),
            "status": "failed",
        }

    finally:
        # GPU 메모리 정리 (다음 청크 처리 안정성)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_model()
    runpod.serverless.start({"handler": handler})
