FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# ✅ 핵심: torch를 먼저 고정 설치 후, 나머지 설치
RUN pip install --no-cache-dir \
    torch==2.4.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .
COPY *.wav .

CMD ["python", "-u", "handler.py"]
