import runpod
import base64
import urllib.request
import io
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
import traceback

print("🚀 [Cold Start] Qwen3-TTS 모델 로딩 중...")
# 1.7B Base 모델 로드 (Zero-shot Voice Cloning용)
# RTX A6000 등 고성능 GPU의 Flash Attention을 활용해 최고 속도를 냅니다.
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa" 
)
print("✅ Qwen3-TTS 로딩 완료! n8n의 명령을 기다립니다.")

def generate_audio(job):
    req = job["input"]
    
    text = req.get("text", "")
    reference_text = req.get("reference_text", "")
    reference_audio_url = req.get("reference_audio", "")
    language = req.get("language", "auto")
    
    print(f"📥 [작업 수신] 대본: {text[:20]}...")
    
    try:
        # 1. n8n이 넘겨준 레퍼런스 오디오(URL)를 임시 폴더에 다운로드
        prompt_audio_path = "/tmp/temp_ref.wav"
        
        # 봇 차단(403 에러) 방지를 위한 헤더 추가
        header = {'User-Agent': 'Mozilla/5.0'}
        request = urllib.request.Request(reference_audio_url, headers=header)
        with urllib.request.urlopen(request) as response, open(prompt_audio_path, 'wb') as out_file:
            out_file.write(response.read())
            
        print("🎙️ 레퍼런스 오디오 다운로드 완료. 음성 복제(추론)를 시작합니다...")
        
        # 2. Qwen-TTS 실제 음성 생성 로직
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=prompt_audio_path,
            ref_text=reference_text
        )
        
        # 3. [서버리스 핵심] 완성된 오디오를 하드디스크에 저장하지 않고, 
        # 곧바로 메모리에서 Base64 텍스트로 압축하여 n8n에 빛의 속도로 쏩니다.
        buffer = io.BytesIO()
        sf.write(buffer, wavs[0], sr, format='WAV')
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
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

# RunPod 서버리스 엔진 가동
runpod.serverless.start({"handler": generate_audio})
