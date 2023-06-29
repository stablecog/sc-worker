import os
import tempfile
import requests
from moviepy.editor import AudioFileClip, VideoClip
from io import BytesIO
from typing import List
import base64
import numpy as np
from scipy.io.wavfile import read as wav_read
import cv2


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


def audio_array_from_wav(wav_bytes: BytesIO, count: int = 50):
    # Read the WAV data
    wav_bytes.seek(0)
    sample_rate, audio_data = wav_read(wav_bytes)

    # Check if audio_data is stereo (2 channels). If yes, convert it to mono
    if len(audio_data.shape) == 2:
        audio_data = np.mean(audio_data, axis=1)

    # Convert to floating point
    audio_data = audio_data.astype(np.float32)

    # Pick two points in each window and average them
    window_size = len(audio_data) // count
    audio_data = audio_data[: window_size * count]  # Discard end samples if necessary
    audio_data = audio_data.reshape(-1, window_size)
    point_values1 = audio_data[:, window_size // 4]
    point_values2 = audio_data[:, window_size * 2 // 4]
    point_values3 = audio_data[:, window_size * 3 // 4]
    avg_values = (point_values1 + point_values2 + point_values3) / 3

    # Convert average values to dB
    avg_db = 20 * np.log10(np.maximum(1e-5, np.abs(avg_values)))

    # Normalize dB values to [0, 1] range
    avg_db_min, avg_db_max = avg_db.min(), avg_db.max()
    audio_data = (avg_db - avg_db_min) / (avg_db_max - avg_db_min)

    audio_data_list = [float(x) for x in audio_data]
    return audio_data_list


def convert_audio_to_video(
    wav_bytes: BytesIO, speaker: str, prompt: str, audio_array: List[float]
) -> BytesIO:
    image_url = get_waveform_image_url(speaker, prompt, audio_array)
    overlay_path = os.path.join(
        os.path.dirname(__file__), "..", "assets", "overlay.png"
    )

    fps = 30

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
        audio_file.write(wav_bytes.getbuffer())
        audio_file_path = audio_file.name

    audioclip = AudioFileClip(audio_file_path)

    response = requests.get(image_url)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as img_file:
        img_file.write(response.content)
        img_file_path = img_file.name

        base_image = cv2.imread(img_file_path)
        moving_image = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
        moving_image_height, moving_image_width, _ = moving_image.shape

    padding = 48
    total_positions = base_image.shape[1] - (2 * padding)

    def make_frame(t):
        new_image = base_image.copy()

        if 1 / fps <= t < audioclip.duration - 1 / fps:
            position = min(
                int((t - 1 / fps) / (audioclip.duration - 2 / fps) * total_positions),
                total_positions,
            )
            position += padding

            for c in range(0, 3):
                new_image[
                    :moving_image_height, position : total_positions + padding, c
                ] = moving_image[:, : total_positions + padding - position, c] * (
                    moving_image[:, : total_positions + padding - position, 3] / 255.0
                ) + new_image[
                    :moving_image_height, position : total_positions + padding, c
                ] * (
                    1.0
                    - moving_image[:, : total_positions + padding - position, 3] / 255.0
                )

        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        return new_image

    imgclip = (
        VideoClip(make_frame, duration=audioclip.duration)
        .set_duration(audioclip.duration)
        .set_fps(fps)
    )

    videoclip = imgclip.set_audio(audioclip)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as output_file:
        videoclip.write_videofile(output_file.name, codec="libx264", audio_codec="aac")
        output_file_path = output_file.name

    with open(output_file_path, "rb") as f:
        output_bytes = BytesIO(f.read())

    print("Audio duration:", audioclip.duration)
    print("Video duration:", imgclip.duration)
    print("Final video duration:", videoclip.duration)

    os.remove(audio_file_path)
    os.remove(img_file_path)
    os.remove(output_file_path)

    return output_bytes
