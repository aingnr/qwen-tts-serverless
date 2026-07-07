import runpod
import base64
import urllib.request
import io
import torch
import numpy as np
import soundfile as sf
import librosa
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


# ─────────────────────────────────────────────
# [FIX 4] 채널별 배속(Tempo) 조정
# Master Config Sheet의 TTS_Tempo 값을 n8n이 전달 → 여기서 적용
# rate < 1.0 → 느려짐 (숨가쁜 속도 문제 보정)
# rate = 1.0 (기본값) → 변경 없음, 미설정 채널 안전장치
# 실패 시 예외를 상위로 전파하여 error 처리 (Option A: 엄격 모드)
# ─────────────────────────────────────────────
def apply_tempo(audio: np.ndarray, tempo: float) -> np.ndarray:
    if tempo == 1.0:
        return audio
    return librosa.effects.time_stretch(audio.astype(np.float32), rate=tempo)


def generate_audio(job):
    req = job["input"]

    text                  = req.get("text", "")
    reference_text        = req.get("reference_text", "")
    reference_audio_url   = req.get("reference_audio", "")
    language              = req.get("language", "auto")
    tempo                 = float(req.get("tempo", 1.0))

    print(f"📥 [작업 수신] 대본: {text[:30]}... / tempo={tempo}")

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

        audio = wavs[0]

        # 배속 조정 (정규화보다 먼저 적용 — 스트레치 후 피크가 달라질 수 있으므로)
        if tempo != 1.0:
            print(f"🎚️ 배속 조정 적용: rate={tempo}")
            audio = apply_tempo(audio, tempo)

        # 정규화 → Base64 인코딩
        normalized = normalize_audio(audio)

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
