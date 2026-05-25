FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# 1. torch 관련 패키지를 먼저 올바른 버전으로 설치
RUN pip install --no-cache-dir \
    torch==2.4.1 torchaudio torchvision \
    --index-url https://download.pytorch.org/whl/cu124

# 2. qwen-tts를 의존성 없이 설치 (torch 덮어쓰기 차단) ✅
RUN pip install --no-cache-dir --no-deps qwen-tts

# 3. 나머지 패키지 설치
RUN pip install --no-cache-dir \
    runpod transformers accelerate soundfile

# 4. torch 버전 확인 (빌드 로그에서 검증용)
RUN python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.version.cuda)"

COPY handler.py .
COPY *.wav .

CMD ["python", "-u", "handler.py"]
