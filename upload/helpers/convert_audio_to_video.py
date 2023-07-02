from pydub import AudioSegment
import cv2
import ffmpeg
from typing import List
import os
import tempfile
import requests
import math
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from upload.helpers.get_waveform_image_url import get_waveform_image_url


def process_frame(
    i,
    total_frames,
    base_image,
    moving_image,
    total_positions,
    padding,
    moving_image_width,
    moving_image_height,
):
    new_image = base_image.copy()
    if i > 0 and i < total_frames - 1:
        position = min(
            int((i - 1) / (total_frames - 2) * total_positions),
            total_positions,
        )
        position += padding

        overlay_width = min(total_positions + padding - position, moving_image_width)

        for c in range(0, 3):
            new_image[
                :moving_image_height, position : position + overlay_width, c
            ] = moving_image[:, :overlay_width, c] * (
                moving_image[:, :overlay_width, 3] / 255.0
            ) + new_image[
                :moving_image_height, position : position + overlay_width, c
            ] * (
                1.0 - moving_image[:, :overlay_width, 3] / 255.0
            )
    return new_image


def convert_audio_to_video(
    wav_bytes: BytesIO, speaker: str, prompt: str, audio_array: List[float]
) -> BytesIO:
    image_url = get_waveform_image_url(speaker, prompt, audio_array)
    overlay_path = os.path.join(
        os.path.dirname(__file__), "../..", "assets", "overlay.png"
    )

    fps = 30

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
        audio_file.write(wav_bytes.getbuffer())
        audio_file_path = audio_file.name

    audio = AudioSegment.from_wav(audio_file_path)
    total_frames = math.ceil(audio.duration_seconds * fps)

    response = requests.get(image_url)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as img_file:
        img_file.write(response.content)
        img_file_path = img_file.name

    base_image = cv2.imread(img_file_path)
    moving_image = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    moving_image_height, moving_image_width, _ = moving_image.shape

    padding = 58
    total_positions = base_image.shape[1] - (2 * padding)

    fourcc = cv2.VideoWriter_fourcc(*"RGBA")
    raw_video_path = audio_file_path.replace(".wav", ".avi")
    output_video_path = audio_file_path.replace(".wav", "_output.mp4")

    video = cv2.VideoWriter(
        raw_video_path,
        fourcc,
        fps,
        (base_image.shape[1], base_image.shape[0]),
    )

    with ThreadPoolExecutor() as executor:
        process_partial = partial(
            process_frame,
            total_frames=total_frames,
            base_image=base_image,
            moving_image=moving_image,
            total_positions=total_positions,
            padding=padding,
            moving_image_width=moving_image_width,
            moving_image_height=moving_image_height,
        )
        for frame in executor.map(process_partial, range(total_frames)):
            video.write(frame)

    video.release()

    output = ffmpeg.output(
        ffmpeg.input(raw_video_path),
        ffmpeg.input(audio_file_path),
        output_video_path,
        **{
            "vcodec": "libx264",  # H.264 codec
            "acodec": "aac",  # Audio codec to be used
            "pix_fmt": "yuv420p",
            "b:a": "320k",
            "crf": "18",
        }
    )
    output.run()

    with open(output_video_path, "rb") as f:
        output_bytes = BytesIO(f.read())

    os.remove(img_file_path)
    os.remove(audio_file_path)
    os.remove(raw_video_path)
    os.remove(output_video_path)

    return output_bytes
