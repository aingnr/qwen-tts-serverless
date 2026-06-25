# ═══════════════════════════════════════════════════════════════
# handler.py — Qwen3-TTS RunPod Serverless Handler
# Version : v2.0
# Updated : 2026-06-25
# Author  : Wizard + Thor
#
# ▣ 이 버전의 수정 목적: 청크별 음질 불일치 문제 해결
#
# [FIX 1] 레퍼런스 오디오 전역 캐싱 (get_reference_audio)
#   - 문제: 청크마다 레퍼런스 오디오를 URL에서 재다운로드
#           → 네트워크 편차·파일 손상으로 Voice Clone 기준점이 흔들림
#   - 해결: Cold Start 시 최초 1회만 다운로드 후 메모리에 캐싱
#           동일 URL 재요청 시 캐시 히트로 즉시 반환
#
# [FIX 2] Peak Normalization (normalize_audio)
#   - 문제: 청크별 볼륨(RMS/Peak)이 달라 이어붙이면 음량 편차가 들림
#   - 해결: 모든 청크 오디오를 Peak 0.95 기준으로 정규화 후 인코딩
#
# [FIX 3] 추론 Seed 고정 (torch.manual_seed)
#   - 문제: LLM 기반 TTS 특성상 매 추론마다 샘플링이 달라짐
#           → 동일 레퍼런스·텍스트여도 청크마다 톤·속도·억양이 변동
#   - 해결: 매 청크 추론 전 FIXED_SEED=42 고정
#           CPU·CUDA 양쪽 seed 모두 세팅하여 완전한 결정론적 추론 보장
# ═══════════════════════════════════════════════════════════════

import runpod
import base64
import urllib.request
import io
import torch
import numpy as np
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
