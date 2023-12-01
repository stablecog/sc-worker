from boto3_type_annotations.s3 import ServiceResource
from PIL import Image
from predict.voiceover.classes import RemoveSilenceParams
from shared.helpers import (
    ensure_trailing_slash,
    parse_content_type,
    convert_wav_to_mp3,
    remove_silence_from_wav,
)
from typing import Any, Dict, Iterable, List
from predict.image.predict import (
    PredictOutput as PredictOutputForImage,
)
from predict.voiceover.predict import (
    PredictOutput as PredictOutputForVoiceover,
)
import time
from io import BytesIO
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
import wave
from pydub import AudioSegment

from upload.helpers.audio_array_from_wav import audio_array_from_wav
from upload.helpers.convert_audio_to_video import convert_audio_to_video
from upload.helpers.get_audio_duration import get_audio_duration


def convert_and_upload_image_to_s3(
    s3: ServiceResource,
    s3_bucket: str,
    pil_image: Image.Image,
    target_quality: int,
    target_extension: str,
    upload_path_prefix: str,
) -> str:
    """Convert an individual image to a target format and upload to S3."""
    start_conv = time.time()

    _pil_image = pil_image
    if target_extension == "jpeg":
        print(f"-- Upload: Converting to JPEG")
        _pil_image = _pil_image.convert("RGB")
    img_format = target_extension.upper()
    img_bytes = BytesIO()
    _pil_image.save(img_bytes, format=img_format, quality=target_quality)
    file_bytes = img_bytes.getvalue()
    end_conv = time.time()
    print(
        f"Converted image in: {round((end_conv - start_conv) *1000)} ms - {img_format} - {target_quality}"
    )

    key = f"{str(uuid.uuid4())}.{target_extension}"
    if upload_path_prefix is not None and upload_path_prefix != "":
        key = f"{ensure_trailing_slash(upload_path_prefix)}{key}"

    content_type = parse_content_type(target_extension)
    start_upload = time.time()
    print(f"-- Upload: Uploading to S3")
    s3.Bucket(s3_bucket).put_object(Body=file_bytes, Key=key, ContentType=content_type)
    end_upload = time.time()
    print(f"Uploaded image in: {round((end_upload - start_upload) *1000)} ms")

    return f"s3://{s3_bucket}/{key}"


def upload_files_for_image(
    uploadObjects: List[PredictOutputForImage],
    s3: ServiceResource,
    s3_bucket: str,
    upload_path_prefix: str,
) -> Iterable[Dict[str, Any]]:
    """Send all files to S3 in parallel and return the S3 URLs"""
    print("Started - Upload all files to S3 in parallel and return the S3 URLs")
    start = time.time()

    # Run all uploads at same time in threadpool
    tasks: List[Future] = []
    with ThreadPoolExecutor(max_workers=len(uploadObjects)) as executor:
        print(f"-- Upload: Submitting to thread")
        for uo in uploadObjects:
            tasks.append(
                executor.submit(
                    convert_and_upload_image_to_s3,
                    s3,
                    s3_bucket,
                    uo.pil_image,
                    uo.target_quality,
                    uo.target_extension,
                    upload_path_prefix,
                )
            )

    # Get results
    results = []
    for i, task in enumerate(tasks):
        print(f"-- Upload: Got result")
        uploadObject = uploadObjects[i]
        results.append(
            {
                "image": task.result(),
                "image_embed": uploadObject.open_clip_image_embed,
                "aesthetic_rating_score": uploadObject.aesthetic_rating_score,
                "aesthetic_artifact_score": uploadObject.aesthetic_artifact_score,
            }
        )

    end = time.time()
    print(
        f"📤 All converted and uploaded to S3 in: {round((end - start) *1000)} ms - Bucket: {s3_bucket} 📤"
    )

    return results


def convert_and_upload_audio_file_to_s3(
    s3: ServiceResource,
    s3_bucket: str,
    audio_bytes: BytesIO,
    remove_silence_params: RemoveSilenceParams,
    sample_rate: int,
    target_extension: str,
    upload_path_prefix: str,
    speaker: str,
    prompt: str,
) -> tuple[str, int]:
    if remove_silence_params.should_remove:
        s = time.time()
        audio_bytes = remove_silence_from_wav(audio_bytes, remove_silence_params)
        e = time.time()
        print(f"🔊 Removed silence in: {round((e - s) *1000)} ms 🔊")
    else:
        audio_segment = AudioSegment.from_wav(audio_bytes)
        audio_segment.export(audio_bytes, format="wav")
        audio_bytes.seek(0)

    audio_duration = get_audio_duration(audio_bytes)
    audio_array = audio_array_from_wav(audio_bytes)

    audio_bytes_converted = None
    s_conv = time.time()
    content_type_audio = "audio/wav"
    if target_extension == "mp3":
        content_type_audio = "audio/mpeg"
        audio_bytes_converted = convert_wav_to_mp3(audio_bytes)
    else:
        audio_bytes_converted = audio_bytes
    e_conv = time.time()
    print(
        f"Converted audio in: {round((e_conv - s_conv) *1000)} ms - {target_extension}"
    )

    s_vid = time.time()
    content_type_video = "video/mp4"
    video_bytes = convert_audio_to_video(
        wav_bytes=audio_bytes,
        speaker=speaker,
        prompt=prompt,
        audio_array=audio_array,
    )
    e_vid = time.time()
    print(f"Created video in: {round((e_vid - s_vid) *1000)} ms")

    new_uuid = str(uuid.uuid4())

    key = f"{new_uuid}.{target_extension}"
    if upload_path_prefix is not None and upload_path_prefix != "":
        key = f"{ensure_trailing_slash(upload_path_prefix)}{key}"
    start_upload = time.time()
    print(f"-- Upload: Uploading to S3")
    s3.Bucket(s3_bucket).put_object(
        Body=audio_bytes_converted, Key=key, ContentType=content_type_audio
    )
    end_upload = time.time()
    print(f"Uploaded audio file in: {round((end_upload - start_upload) *1000)} ms")
    audio_url = f"s3://{s3_bucket}/{key}"

    key_video = f"{new_uuid}.mp4"
    if upload_path_prefix is not None and upload_path_prefix != "":
        key_video = f"{ensure_trailing_slash(upload_path_prefix)}{key_video}"
    start_upload = time.time()
    print(f"-- Upload: Uploading to S3")
    s3.Bucket(s3_bucket).put_object(
        Body=video_bytes, Key=key_video, ContentType=content_type_video
    )
    end_upload = time.time()
    print(f"Uploaded video file in: {round((end_upload - start_upload) *1000)} ms")
    video_url = f"s3://{s3_bucket}/{key_video}"

    return [audio_url, audio_duration, video_url, audio_array]


def upload_files_for_voiceover(
    uploadObjects: List[PredictOutputForVoiceover],
    s3: ServiceResource,
    s3_bucket: str,
    upload_path_prefix: str,
) -> Iterable[Dict[str, Any]]:
    """Send all files to S3 in parallel and return the S3 URLs"""
    print("Started - Upload all files to S3 in parallel and return the S3 URLs")
    start = time.time()

    # Run all uploads at same time in threadpool
    tasks: List[Future] = []
    with ThreadPoolExecutor(max_workers=len(uploadObjects)) as executor:
        print(f"-- Upload: Submitting to thread")
        for uo in uploadObjects:
            tasks.append(
                executor.submit(
                    convert_and_upload_audio_file_to_s3,
                    s3,
                    s3_bucket,
                    uo.audio_bytes,
                    uo.remove_silence_params,
                    uo.sample_rate,
                    uo.target_extension,
                    upload_path_prefix,
                    uo.speaker,
                    uo.prompt,
                )
            )

    # Get results
    results = []
    for task in tasks:
        print(f"-- Upload: Got result")
        [
            audio_file,
            audio_duration,
            video_file,
            audio_array,
        ] = task.result()
        results.append(
            {
                "audio_file": audio_file,
                "audio_duration": audio_duration,
                "video_file": video_file,
                "audio_array": audio_array,
            }
        )

    end = time.time()
    print(
        f"📤 All converted and uploaded to S3 in: {round((end - start) *1000)} ms - Bucket: {s3_bucket} 📤"
    )

    return results
