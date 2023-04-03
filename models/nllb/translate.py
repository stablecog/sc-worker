from lingua import Language, LanguageDetector
import time
from .constants import (
    LANG_TO_FLORES,
    TARGET_LANG,
    TARGET_LANG_FLORES,
    TARGET_LANG_SCORE_MAX,
)
from transformers import pipeline
import torch
from typing import Any


def translate_prompt_set(
    text_1: str,
    flores_1: str | None,
    text_2: str,
    flores_2: str | None,
    translator: Any,
    label: str,
):
    startTimeTranslation = time.time()

    translated_text_1 = translate_text(
        text=text_1,
        flores=flores_1,
        label=f"{label} - #1",
        translator=translator,
    )
    translated_text_2 = translate_text(
        text=text_2,
        flores=flores_2,
        label=f"{label} - #1",
        translator=translator,
    )

    endTimeTranslation = time.time()
    print(
        f"-- {label} - Translation completed in: {round((endTimeTranslation - startTimeTranslation) * 1000)} ms --"
    )

    return [translated_text_1, translated_text_2]


def translate_text(
    text: str | None,
    flores: str | None,
    translator: Any,
    label: str,
):
    start = time.time()
    if text == "":
        print(f"-- {label} - No text to translate, skipping --")
        return ""

    detected_flores = TARGET_LANG_FLORES
    translated_text = ""
    detected_flores = get_flores(
        text=text,
        flores=flores,
        detector=translator["detector"],
        label=label,
    )

    if detected_flores == TARGET_LANG_FLORES:
        translated_text = text
        print(
            f"-- {label} - Text is already in the correct language, no translation needed --"
        )
        print(f'-- {label} - #1 - Text is: "{translated_text}" --')
    else:
        translate = pipeline(
            "translation",
            model=translator["model"],
            tokenizer=translator["tokenizer"],
            torch_dtype=torch.float16,
            src_lang=detected_flores,
            tgt_lang=TARGET_LANG_FLORES,
            device=0,
        )
        translate_output = translate(text, max_length=1000)
        translated_text = translate_output[0]["translation_text"]
        print(f'-- {label} - Original text is: "{text}" --')
        print(f'-- {label} - Translated text is: "{translated_text}" --')

    end = time.time()
    print(f"-- {label} - Translated text in: {round((end - start) * 1000)} ms --")
    return translated_text


def get_flores(
    text,
    flores,
    detector,
    label,
):
    if text == "":
        print(f"-- {label} - No text to give FLORES-200 for, skipping --")
        return TARGET_LANG_FLORES
    if flores is not None and flores is not "":
        print(
            f'-- {label} - FLORES-200 code is given, skipping language auto-detection: "{flores}" --'
        )
        return flores

    text_lang_flores = TARGET_LANG_FLORES
    confidence_values = detector.compute_language_confidence_values(text)
    target_lang_score = None
    detected_lang = None
    detected_lang_score = None

    print(f"-- Confidence values - {confidence_values[:10]} --")
    for index in range(len(confidence_values)):
        curr = confidence_values[index]
        if index == 0:
            detected_lang = curr[0]
            detected_lang_score = curr[1]
        if curr[0] == Language.ENGLISH:
            target_lang_score = curr[1]

    if (
        detected_lang is not None
        and detected_lang != TARGET_LANG
        and (target_lang_score is None or target_lang_score < TARGET_LANG_SCORE_MAX)
        and LANG_TO_FLORES.get(detected_lang.name) is not None
    ):
        text_lang_flores = LANG_TO_FLORES[detected_lang.name]

    if detected_lang is not None:
        print(
            f'-- {label} - Guessed text language: "{detected_lang.name}". Score: {detected_lang_score} --'
        )
    if (
        detected_lang is not None
        and target_lang_score is not None
        and detected_lang != TARGET_LANG
    ):
        print(f"-- {label} - Target language score: {target_lang_score} --")

    print(f'-- {label} - Selected text language FLORES-200: "{text_lang_flores}" --')
    return text_lang_flores
