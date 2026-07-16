# 1. 우분투와 CUDA(GPU)가 깔린 베이스 이미지를 가져옵니다.
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
# 2. 작업 폴더 지정
WORKDIR /workspace
# 3. 오디오 처리에 필수인 ffmpeg 설치
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
# 4. 파이썬 부품(requirements.txt) 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 모델 캐시 경로 고정 (빌드 타임 다운로드 + 런타임 로드가 같은 경로를 보도록)
ENV HF_HOME=/workspace/hf_cache

# 6. 모델을 빌드 타임에 이미지 안으로 미리 다운로드 (GPU 불필요, 파일만 받음)
#    ARG CACHE_BUST: 값을 바꿔서 push하면 이 레이어를 강제로 재실행 (Docker 레이어 캐싱 우회)
ARG CACHE_BUST=1
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base')"

# 7. 런타임 오프라인 모드 고정 — 캐시된 모델만 사용, HF 네트워크 콜 완전 차단
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# 8. 우리가 만든 심장 엔진 복사
COPY handler.py .
# 9. 컨테이너가 켜지면 handler.py를 실행하라!
CMD ["python", "-u", "handler.py"]
