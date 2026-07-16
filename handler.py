import runpod
import base64
import re
import urllib.request
import io
import torch
import numpy as np
import soundfile as sf
from huggingface_hub import snapshot_download
from qwen_tts import Qwen3TTSModel
import traceback

print("🚀 [Cold Start] Qwen3-TTS 모델 로딩 중...")

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

# 이미 Docker 이미지 빌드 타임에 캐싱되어 있으므로, 네트워크 호출 없이
# 로컬 캐시 경로만 즉시 반환됨. 이 경로를 from_pretrained에 넘기면
# transformers 내부의 "저장소 ID 문자열일 때만 네트워크를 타는" 버그
# 지점(_patch_mistral_regex → is_base_mistral)을 완전히 우회한다.
local_model_path = snapshot_download(MODEL_ID, local_files_only=True)

model = Qwen3TTSModel.from_pretrained(
    local_model_path,
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
    local_files_only=True,
)

print("✅ Qwen3-TTS 로딩 완료! n8n의 명령을 기다립니다.")

# ─────────────────────────────────────────────
# [FIX 1] 레퍼런스/청크 오디오 전역 캐시
# Cold Start 이후 동일 URL은 재다운로드하지 않음
# → voice_clone의 레퍼런스 오디오, merge의 청크 오디오 공용 재사용
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
        print(f"📥 오디오 캐싱 완료: {path}")
    else:
        print(f"✅ 오디오 캐시 히트: {REF_AUDIO_CACHE[url]}")
    return REF_AUDIO_CACHE[url]


# ─────────────────────────────────────────────
# [FIX 2] Peak Normalization
# 청크별/병합본 볼륨 편차를 제거하여 균일한 음량 보장
# ─────────────────────────────────────────────
def normalize_audio(audio: np.ndarray) -> np.ndarray:
    audio = audio.astype(np.float32)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95
    return audio


# ─────────────────────────────────────────────
# [FIX 6] Fade In/Out
# 청크 시작/끝 지점의 파형 불연속(클릭/팝 노이즈) 방지.
# 무음을 붙이기 전 반드시 이 함수를 먼저 적용한다.
# ─────────────────────────────────────────────
def apply_fade(audio: np.ndarray, sr: int, fade_ms: int = 8) -> np.ndarray:
    fade_len = int(sr * fade_ms / 1000)
    if fade_len <= 0 or len(audio) < fade_len * 2:
        return audio
    audio = audio.copy()
    fade_in = np.linspace(0.0, 1.0, fade_len, dtype=audio.dtype)
    fade_out = np.linspace(1.0, 0.0, fade_len, dtype=audio.dtype)
    audio[:fade_len] *= fade_in
    audio[-fade_len:] *= fade_out
    return audio


# ─────────────────────────────────────────────
# [FIX 5] 청크 오디오 병합 (Merge)
# Part_N_[ID].wav "파일명"에서 숫자를 추출해 정렬 후,
# silence_gap(초) 만큼 무음을 사이에 넣어 이어붙임.
# Google Drive 다운로드 URL(예: uc?export=download&id=...)에는
# 파일명이 포함되지 않으므로, name과 url을 쌍으로 받아
# name 기준으로 정렬하고 url로 다운로드한다.
# 개별 청크는 재정규화하지 않고(이미 정규화된 상태),
# 최종 병합본에만 한 번 더 peak normalize를 적용한다.
# 샘플레이트 불일치·패턴 불일치 시 예외를 던져 A안(엄격) 처리.
# ─────────────────────────────────────────────
def extract_part_num(name: str) -> int:
    m = re.search(r"Part_(\d+)_", name)
    if not m:
        raise ValueError(f"Part 번호를 파일명에서 찾을 수 없음: {name}")
    return int(m.group(1))


def merge_chunks(chunks: list, silence_gap: float = 0.5):
    if not chunks:
        raise ValueError("chunks가 비어 있습니다.")

    for c in chunks:
        if "name" not in c or "url" not in c:
            raise ValueError(f"chunks 항목에 name/url이 없습니다: {c}")

    sorted_chunks = sorted(chunks, key=lambda c: extract_part_num(c["name"]))

    segments = []
    sr_ref = None
    for c in sorted_chunks:
        path = get_reference_audio(c["url"])
        audio, sr = sf.read(path)
        if sr_ref is None:
            sr_ref = sr
        elif sr != sr_ref:
            raise ValueError(f"샘플레이트 불일치: {c['name']} ({sr} != {sr_ref})")
        segments.append(audio)

    gap = np.zeros(int(silence_gap * sr_ref), dtype=segments[0].dtype)
    merged = segments[0]
    for seg in segments[1:]:
        merged = np.concatenate([merged, gap, seg])

    return merged, sr_ref


def generate_audio(job):
    req = job["input"]
    mode = req.get("mode", "voice_clone")

    # ─────────────────────────────────────────────
    # MODE: merge — 청크 오디오 병합
    # ─────────────────────────────────────────────
    if mode == "merge":
        chunks = req.get("chunks", [])
        silence_gap = float(req.get("silence_gap", 0.5))
        print(f"📥 [작업 수신] 병합 모드 / 청크 수={len(chunks)} / silence_gap={silence_gap}s")

        try:
            merged_audio, sr = merge_chunks(chunks, silence_gap)

            # 최종 병합본 전체 피크 정규화 — 청크별 음량 편차 해소
            merged_audio = normalize_audio(merged_audio)

            buffer = io.BytesIO()
            sf.write(buffer, merged_audio, sr, format="WAV")
            buffer.seek(0)
            audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")

            print("✨ 병합 및 Base64 변환 성공!")
            return {
                "status": "success",
                "message": "청크 병합 완료!",
                "audio_base64": audio_base64
            }

        except Exception as e:
            print(f"❌ 병합 중 에러 발생: {e}")
            traceback.print_exc()
            return {"error": str(e)}

    # ─────────────────────────────────────────────
    # MODE: voice_clone (기본값) — tempo 기능 제거, 원본 속도 고정
    # ─────────────────────────────────────────────
    text                  = req.get("text", "")
    reference_text        = req.get("reference_text", "")
    reference_audio_url   = req.get("reference_audio", "")
    language              = req.get("language", "auto")
    chunk_gap             = float(req.get("chunk_gap", 0.4))
    seed                  = int(req.get("seed", 42))

    print(f"📥 [작업 수신] 대본: {text[:30]}... / chunk_gap={chunk_gap}s / seed={seed}")

    try:
        # ─────────────────────────────────────────────
        # [FIX 3] Seed — 기본 42 (최초 생성, 톤 일관성 유지)
        # → 재시도 시 n8n이 랜덤 시드를 보내면 그 값을 그대로 사용,
        #   같은 텍스트에서 반복적으로 같은 불량 결과가 재현되는 문제 방지
        # ─────────────────────────────────────────────
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # 레퍼런스 오디오: 캐시에서 가져오기 (최초 1회만 다운로드)
        prompt_audio_path = get_reference_audio(reference_audio_url)

        print("🎙️ 음성 복제(추론) 시작...")

        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=prompt_audio_path,
            ref_text=reference_text
        )

        # 정규화 → Fade In/Out(클릭 노이즈 방지) → 끝에 무음 삽입
        normalized = normalize_audio(wavs[0])
        normalized = apply_fade(normalized, sr)

        if chunk_gap > 0:
            silence = np.zeros(int(chunk_gap * sr), dtype=normalized.dtype)
            normalized = np.concatenate([normalized, silence])

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
