import wave
from io import BytesIO


def get_audio_duration(audio_bytes: BytesIO) -> int:
    # Make sure we're at the start of the BytesIO object
    audio_bytes.seek(0)

    # Open the BytesIO object as a WAV file
    wav_file = wave.open(audio_bytes, "rb")

    # Extract the necessary information
    bit_depth = wav_file.getsampwidth() * 8  # bit_depth in bits
    num_channels = wav_file.getnchannels()
    sample_rate = wav_file.getframerate()

    # Calculate the number of samples
    num_bytes = len(audio_bytes.getvalue())
    num_samples = (
        num_bytes * 8 / bit_depth
    )  # number of samples = total bits / bits per sample

    # If the audio is stereo (2 channels), divide the number of samples by 2
    if num_channels == 2:
        num_samples = num_samples // 2

    # Calculate the duration
    duration = num_samples / sample_rate

    return duration
