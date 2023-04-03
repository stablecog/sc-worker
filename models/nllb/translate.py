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
    flores_code_1: str | None,
    text_2: str,
    flores_code_2: str | None,
    translator: Any,
    label: str,
):
    startTimeTranslation = time.time()

    args = {
        "translator": translator,
        "target_flores": TARGET_LANG_FLORES,
    }
    translated_text_1 = translate_text(
        text=text_1,
        text_flores=flores_code_1,
        label=f"{label} - #1",
        **args,
    )
    translated_text_2 = translate_text(
        text=text_2,
        text_flores=flores_code_2,
        label=f"{label} - #1",
        **args,
    )

    endTimeTranslation = time.time()
    print(
        f"-- {label} - Translation completed in: {round((endTimeTranslation - startTimeTranslation) * 1000)} ms --"
    )

    return [translated_text_1, translated_text_2]


def translate_text(
    text: str | None,
    text_flores: str | None,
    translator: Any,
    label: str,
):
    if text == "":
        print(f"-- {label} - No text to translate, skipping --")
        return ""

    text_lang_flores = TARGET_LANG_FLORES
    translated_text = ""
    text_lang_flores = get_flores_200_code(
        text=text,
        defined_flores_code=text_flores,
        detector=translator["detector"],
        label=label,
    )

    if text_lang_flores == TARGET_LANG_FLORES:
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
            src_lang=text_flores,
            tgt_lang=TARGET_LANG_FLORES,
            device=0,
        )
        translate_output = translate(text, max_length=1000)
        translated_text = translate_output[0]["translation_text"]
        print(f'-- {label} - Original text is: "{text}" --')
        print(f'-- {label} - Translated text is: "{translated_text}" --')

    return translated_text


def get_flores_200_code(
    text,
    defined_flores_code,
    detector,
    label,
):
    if text == "":
        print(f"-- {label} - No text to give FLORES-200 for, skipping --")
        return TARGET_LANG_FLORES
    if defined_flores_code is not None and defined_flores_code is not "":
        print(
            f'-- {label} - FLORES-200 code is given, skipping language auto-detection: "{defined_flores_code}" --'
        )
        return defined_flores_code

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
