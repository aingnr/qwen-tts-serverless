import runpod
import base64
import urllib.request
import io
import torch
import numpy as np
import librosa
import soundfile as sf
from qwen_tts import Qwen3TTSModel
import traceback

print("🚀 [Cold Start] Qwen3-TTS 모델 로딩 중...")

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa"
)

print("✅ Qwen3-TTS 로딩 완료! n8n의 명령을 기다립니다.")

# ─────────────────────────────────────────────
# [FIX 1] 레퍼런스 오디오 전역 캐시
# Cold Start 이후 동일 URL은 재다운로드하지 않음
# → 청크마다 네트워크 편차·파일 손상 위험 제거
# ─────────────────────────────────────────────
REF_AUDIO_CACHE = {}

def get_reference_audio(url: str) -> str:
    if url not in REF_AUDIO_CACHE:
        path = f"/tmp/ref_{abs(hash(url))}.wav"
        header = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(url, headers=header)
        with urllib.request.urlopen(req) as resp, open(path, "wb") as f:
            f.write(resp.read())

        # ─────────────────────────────────────────────
        # [FIX 4] 레퍼런스 오디오 클리핑 자동 정규화
        # peak > 1.0 감지 시 0.95 기준으로 정규화 후 덮어쓰기
        # → Qwen3-TTS WARNING 제거 + Voice Clone 기준점 안정화
        # ─────────────────────────────────────────────
        audio, sr = librosa.load(path, sr=None, mono=True)
        peak = np.max(np.abs(audio))
        if peak > 1.0:
            audio = audio / peak * 0.95
            sf.write(path, audio, sr)
            print(f"⚠️ 레퍼런스 오디오 클리핑 감지 → 정규화 적용 (peak: {peak:.4f})")

        REF_AUDIO_CACHE[url] = path
        print(f"📥 레퍼런스 오디오 캐싱 완료: {path}")
    else:
        print(f"✅ 레퍼런스 오디오 캐시 히트: {REF_AUDIO_CACHE[url]}")
    return REF_AUDIO_CACHE[url]


# ─────────────────────────────────────────────
# [FIX 2] Peak Normalization
# 청크별 볼륨 편차를 제거하여 이어붙일 때 균일한 음량 보장
# ─────────────────────────────────────────────
def normalize_audio(audio: np.ndarray) -> np.ndarray:
    audio = audio.astype(np.float32)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95
    return audio


def generate_audio(job):
    req = job["input"]

    text                  = req.get("text", "")
    reference_text        = req.get("reference_text", "")
    reference_audio_url   = req.get("reference_audio", "")
    language              = req.get("language", "auto")

    print(f"📥 [작업 수신] 대본: {text[:30]}...")

    try:
        # ─────────────────────────────────────────────
        # [FIX 3] Seed 고정 — 매 청크 추론 전에 항상 동일한 시드 세팅
        # → 샘플링 랜덤성을 제거하여 청크 간 톤·속도·억양 일관성 확보
        # ─────────────────────────────────────────────
        FIXED_SEED = 42
        torch.manual_seed(FIXED_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FIXED_SEED)

        # 레퍼런스 오디오: 캐시에서 가져오기 (최초 1회만 다운로드)
        prompt_audio_path = get_reference_audio(reference_audio_url)

        print("🎙️ 음성 복제(추론) 시작...")

        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=prompt_audio_path,
            ref_text=reference_text
        )

        # 정규화 → Base64 인코딩
        normalized = normalize_audio(wavs[0])

        buffer = io.BytesIO()
        sf.write(buffer, normalized, sr, format="WAV")
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        print("✨ 렌더링 및 Base64 변환 성공!")
        return {
            "status": "success",
            "message": "서버리스 렌더링 완료!",
            "audio_base64": audio_base64
        }

    except Exception as e:
        print(f"❌ 렌더링 중 에러 발생: {e}")
        traceback.print_exc()
        return {"error": str(e)}


runpod.serverless.start({"handler": generate_audio})
