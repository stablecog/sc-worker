from io import BytesIO
import numpy as np
from scipy.io.wavfile import read as wav_read


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
