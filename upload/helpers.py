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


def audio_array_from_wav(wav_bytes: BytesIO, count: int = 50, overlap: float = 0.1):
    # Read the WAV data
    wav_bytes.seek(0)
    sample_rate, audio_data = wav_read(wav_bytes)

    # Check if audio_data is stereo (2 channels). If yes, convert it to mono
    if len(audio_data.shape) == 2:
        audio_data = np.mean(audio_data, axis=1)

    # Convert to floating point
    audio_data = audio_data.astype(np.float32)

    # Compute RMS over overlapping windows
    window_size = len(audio_data) // count
    stride = int(window_size * (1 - overlap))
    rms_values = []
    for i in range(0, len(audio_data) - window_size, stride):
        window = audio_data[i : i + window_size]
        window = window * np.hanning(window_size)  # Apply Hanning window
        rms = np.sqrt(np.mean(window**2))
        rms_values.append(rms)

    # If we ended up with too many windows due to the overlap, just take the first 'count' ones
    rms_values = rms_values[:count]

    # Convert RMS values to dB, adding a small value to avoid log of zero
    rms_db = 20 * np.log10(np.maximum(1e-5, np.array(rms_values)))

    # Normalize dB values to [0, 1] range, using a standard dB range for audio
    rms_db_min, rms_db_max = -60.0, 0.0
    audio_data = (rms_db - rms_db_min) / (rms_db_max - rms_db_min)

    audio_data_list = [float(x) for x in audio_data]
    return audio_data_list


def convert_audio_to_video(
    wav_bytes: BytesIO, speaker: str, prompt: str, audio_array: List[float]
) -> BytesIO:
    image_url = get_waveform_image_url(speaker, prompt, audio_array)
    cursor_path = os.path.join(os.path.dirname(__file__), "..", "assets", "cursor.png")

    fps = 30

    # write audio bytes to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
        audio_file.write(wav_bytes.getbuffer())
        audio_file_path = audio_file.name

    audioclip = AudioFileClip(audio_file_path)

    # download the image and save it to a temporary file
    response = requests.get(image_url)
    response.raise_for_status()  # ensure we downloaded the image successfully
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as img_file:
        img_file.write(response.content)
        img_file_path = img_file.name
        base_image = cv2.imread(img_file_path)
        moving_image = cv2.imread(
            cursor_path, cv2.IMREAD_UNCHANGED
        )  # include the alpha channel
        moving_image_height, moving_image_width, _ = moving_image.shape

    padding = 48 - moving_image_width // 2

    # Total number of positions for the moving image (subtract one more to ensure the moving image reaches the end)
    total_positions = base_image.shape[1] - moving_image_width - (2 * padding) - 1

    def make_frame(t):
        # Create a new image with the moving image at the correct position
        new_image = base_image.copy()
        # Don't show the moving image in the first and last frame
        if t > 1 / fps and t < audioclip.duration - 1 / fps:
            # Calculate the position of the moving image in this frame
            position = min(
                int((t - 1 / fps) / (audioclip.duration - 2 / fps) * total_positions),
                total_positions,
            )

            # Adjust position by the padding
            position += padding

            # Apply moving image onto new_image while preserving transparency
            for c in range(0, 3):
                new_image[
                    :moving_image_height, position : position + moving_image_width, c
                ] = moving_image[:, :, c] * (moving_image[:, :, 3] / 255.0) + new_image[
                    :moving_image_height, position : position + moving_image_width, c
                ] * (
                    1.0 - moving_image[:, :, 3] / 255.0
                )

        # Convert BGR to RGB
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)

        return new_image

    # Create a clip from the frames
    imgclip = (
        VideoClip(make_frame, duration=audioclip.duration)
        .set_duration(audioclip.duration)
        .set_fps(fps)
    )

    # Set the audio of the video to be the wav file
    videoclip = imgclip.set_audio(audioclip)

    # write the final clip to another temporary file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as output_file:
        videoclip.write_videofile(output_file.name, codec="libx264", audio_codec="aac")
        output_file_path = output_file.name

    # read the temporary output file into a BytesIO object
    with open(output_file_path, "rb") as f:
        output_bytes = BytesIO(f.read())

    # Delete the temporary files
    os.remove(audio_file_path)
    os.remove(img_file_path)
    os.remove(output_file_path)

    return output_bytes
