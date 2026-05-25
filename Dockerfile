# 최신 PyTorch + CUDA 12.4로 업그레이드 (sm_89/sm_90 완벽 지원)
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

# 작업 폴더 지정
WORKDIR /workspace

# 오디오 처리에 필수인 ffmpeg 설치
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# 파이썬 부품(requirements.txt) 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 우리가 만든 심장 연관 복사
COPY handler.py .
COPY *.wav .

# 컨테이너가 커지면 handler.py를 실행하라!
CMD ["python", "-u", "handler.py"]
