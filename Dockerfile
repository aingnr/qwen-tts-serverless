# ── Base Image ──────────────────────────────────────────────────────────────
# pytorch/pytorch:2.4.1-cuda12.4 → sm_86/sm_89/sm_90 GPU 완벽 지원
# (RTX 3090/4090/6000 Ada / H100 모두 커버)
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

# ── System Dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Working Directory ────────────────────────────────────────────────────────
WORKDIR /app

# ── Step 1: qwen-tts 설치 (torch 의존성 덮어쓰기 허용) ──────────────────────
# qwen-tts는 torch를 내부적으로 재설치할 수 있음 → 마지막에 강제 복원
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Step 2: Flash Attention 2 설치 (공식 예제 권장) ─────────────────────────
# 설치 실패 시 handler.py 에서 sdpa 자동 폴백
RUN pip install flash-attn --no-build-isolation || echo "[WARN] flash-attn 설치 실패 → sdpa 폴백 사용"

# ── Step 3: torch 최종 강제 복원 (cu124 버전 보장) ──────────────────────────
# qwen-tts 또는 flash-attn 설치 중 torch 버전이 변경됐을 경우 복원
RUN pip install --no-cache-dir --force-reinstall \
    torch==2.4.1 torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# ── Pre-download Model (빌드 타임 베이크 → 콜드스타트 최소화) ─────────────────
# Voice Clone = Base 모델 + Tokenizer 두 가지 모두 필요
ENV HF_HOME=/app/models

RUN python -c "\
from huggingface_hub import snapshot_download; \
print('[Download] Qwen3-TTS-Tokenizer-12Hz...'); \
snapshot_download('Qwen/Qwen3-TTS-Tokenizer-12Hz', cache_dir='/app/models'); \
print('[Download] Qwen3-TTS-12Hz-1.7B-Base...'); \
snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base', cache_dir='/app/models'); \
print('[Download] Complete.')"

# ── Reference Audio Files (채널별 .wav — URL 폴백용 로컬 복사) ───────────────
# generate_voice_clone()은 URL 직접 지원하므로 로컬 파일은 폴백 전용
# 파일명 규칙: {Channel_ID}_7.wav
# (GoldMan_7.wav · REPCH_7.wav · MYPHARM_7.wav · EcoDIVE_7.wav)
RUN mkdir -p /app/ref_audio
COPY *.wav /app/ref_audio/

# ── Handler ──────────────────────────────────────────────────────────────────
COPY handler.py .

# ── Entrypoint ───────────────────────────────────────────────────────────────
CMD ["python", "-u", "handler.py"]
