# 1. 우분투와 CUDA(GPU)가 깔린 베이스 이미지를 가져옵니다.
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# 2. 작업 폴더 지정
WORKDIR /workspace

# 3. 오디오 처리에 필수인 ffmpeg 설치
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# 4. 파이썬 부품(requirements.txt) 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 우리가 만든 심장 엔진 복사
COPY handler.py .

# 6. 컨테이너가 켜지면 handler.py를 실행하라!
CMD ["python", "-u", "handler.py"]
