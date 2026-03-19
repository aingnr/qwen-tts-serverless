import runpod
import base64
import urllib.request
import io
import os  # 파일 삭제를 위해 추가
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
import traceback

print("🚀 [Cold Start] Qwen3-TTS 모델 로딩 중...")
# 1.7B Base 모델 로드 (Zero-shot Voice Cloning용)
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa" 
)
print("✅ Qwen3-TTS 로딩 완료! n8n의 명령을 기다립니다.")

def generate_audio(job):
    req = job["input"]
    job_id = job.get("id", "local_test") # 🌟 고유 작업 ID 가져오기
    
    text = req.get("text", "")
    reference_text = req.get("reference_text", "")
    reference_audio_url = req.get("reference_audio", "")
    language = req.get("language", "auto")
    
    print(f"📥 [작업 수신] {job_id} | 대본: {text[:20]}...")
    
    # 🌟 고유한 임시 파일명 생성 (충돌 완벽 방지)
    prompt_audio_path = f"/tmp/temp_ref_{job_id}.wav"
    
    try:
        # 1. n8n이 넘겨준 레퍼런스 오디오 다운로드
        header = {'User-Agent': 'Mozilla/5.0'}
        request = urllib.request.Request(reference_audio_url, headers=header)
        with urllib.request.urlopen(request) as response, open(prompt_audio_path, 'wb') as out_file:
            out_file.write(response.read())
            
        print("🎙️ 레퍼런스 오디오 다운로드 완료. 음성 복제를 시작합니다...")
        
        # 2. Qwen-TTS 실제 음성 생성 로직
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=prompt_audio_path,
            ref_text=reference_text
        )
        
        # 3. [서버리스 핵심] 메모리에서 Base64 텍스트로 압축
        buffer = io.BytesIO()
        
        # (안전장치) 모델 결과물이 PyTorch Tensor일 경우를 대비해 Numpy/CPU로 확실히 내려줍니다.
        audio_data = wavs[0].cpu().numpy() if torch.is_tensor(wavs[0]) else wavs[0]
        
        sf.write(buffer, audio_data, sr, format='WAV')
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        print(f"✨ [{job_id}] 렌더링 및 Base64 변환 성공!")
        
        # 4. n8n으로 결과 리턴 (RunPod이 자동으로 "status": "COMPLETED"를 최상단에 붙여줍니다)
        return {
            "audio_base64": audio_base64 
        }
        
    except Exception as e:
        print(f"❌ [{job_id}] 렌더링 중 에러 발생: {e}")
        traceback.print_exc()
        return {"error": str(e)}
        
    finally:
        # 🌟 [매우 중요] 작업이 성공하든 실패하든, 썼던 쓰레기(임시 파일)는 무조건 청소합니다!
        if os.path.exists(prompt_audio_path):
            os.remove(prompt_audio_path)
            print(f"🧹 임시 레퍼런스 파일 청소 완료: {prompt_audio_path}")

# RunPod 서버리스 엔진 가동
runpod.serverless.start({"handler": generate_audio})
