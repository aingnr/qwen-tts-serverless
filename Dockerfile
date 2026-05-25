FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 1. 먼저 qwen-tts 등 나머지 패키지 설치 (torch 덮어써도 OK)
RUN pip install --no-cache-dir -r requirements.txt

# 2. 마지막에 torch를 강제로 올바른 버전으로 덮어씌우기 ✅
RUN pip install --no-cache-dir --force-reinstall \
    torch==2.4.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

COPY handler.py .
COPY *.wav .

CMD ["python", "-u", "handler.py"]
