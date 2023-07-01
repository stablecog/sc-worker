from typing import List
import base64
from io import BytesIO


def encode_string_as_base64(string: str) -> str:
    input_bytes = string.encode("utf-8")
    encoded_bytes = base64.b64encode(input_bytes)
    return encoded_bytes.decode("utf-8")


def get_waveform_image_url(
    speaker: str,
    prompt: str,
    audio_array: List[float],
) -> BytesIO:
    encoded_speaker = encode_string_as_base64(speaker)
    encoded_prompt = encode_string_as_base64(prompt[:200])
    audio_array_string = ",".join([str(x) for x in audio_array])
    return f"https://og.stablecog.com/api/voiceover/waveform.png?speaker={encoded_speaker}&prompt={encoded_prompt}=&audio_array={audio_array_string}"
