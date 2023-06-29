import os
import tempfile
import requests
from moviepy.editor import AudioFileClip, ImageClip
from io import BytesIO
from typing import List
import base64
import numpy as np
from scipy.io.wavfile import read as wav_read


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


def audio_array_from_wav(wav_bytes: BytesIO, count: int):
    # Read the WAV data
    wav_bytes.seek(0)
    sample_rate, audio_data = wav_read(wav_bytes)

    # Check if audio_data is stereo (2 channels). If yes, convert it to mono
    if len(audio_data.shape) == 2:
        audio_data = np.mean(audio_data, axis=1)

    # Convert to floating point
    audio_data = audio_data.astype(np.float32)

    # Compute RMS over windows
    window_size = len(audio_data) // count
    audio_data = audio_data[: window_size * count]  # Discard end samples if necessary
    audio_data = audio_data.reshape(-1, window_size)
    rms_values = np.sqrt(np.mean(audio_data**2, axis=1))

    # Convert RMS values to dB
    rms_db = 20 * np.log10(rms_values)

    # Normalize dB values to [0, 1] range
    rms_db_min, rms_db_max = rms_db.min(), rms_db.max()
    audio_data = (rms_db - rms_db_min) / (rms_db_max - rms_db_min)

    audio_data_list = [float(x) for x in audio_data]
    return audio_data_list


def convert_audio_to_video(
    wav_bytes: BytesIO, speaker: str, prompt: str, audio_array: List[float]
) -> BytesIO:
    image_url = get_waveform_image_url(speaker, prompt, audio_array)

    # write audio bytes to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
        audio_file.write(wav_bytes.getbuffer())
        audio_file_path = audio_file.name

    # download the image and save it to a temporary file
    response = requests.get(image_url)
    response.raise_for_status()  # ensure we downloaded the image successfully
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as img_file:
        img_file.write(response.content)
        img_file_path = img_file.name

    # read the temporary audio file into MoviePy
    audioclip = AudioFileClip(audio_file_path)

    # read the downloaded image file into MoviePy
    imgclip = ImageClip(img_file_path).set_duration(audioclip.duration).set_fps(1)

    # set the audio of the video to be the mp3 file
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
