from lingua import Language, LanguageDetector
from threading import Lock
import time
from .constants import (
    LANG_TO_FLORES,
    TARGET_LANG,
    TARGET_LANG_SCORE_MAX,
    TARGET_LANG_FLORES,
    DETECTED_CONFIDENCE_SCORE_MIN,
)
from transformers import pipeline
import torch
from typing import Any
import requests
from tabulate import tabulate

translator_mutex = Lock()


def translate_text_set_via_api(
    text_1: str,
    flores_1: str | None,
    text_2: str,
    flores_2: str | None,
    translator_url: str,
    detector: LanguageDetector,
    label: str,
):
    with translator_mutex:
        print(f"-- {label} - Translator url is: '{translator_url}' --")

        if text_1 is None:
            text_1 = ""
        if text_2 is None:
            text_2 = ""

        if text_1 == "" and text_2 == "":
            print(f"-- {label} - No text to translate, skipping --")
            return ["", ""]

        startTimeTranslation = time.time()

        translated_text_1 = ""
        translated_text_2 = ""

        text_flores_1 = TARGET_LANG_FLORES
        text_flores_2 = TARGET_LANG_FLORES

        text_flores_1 = get_flores(
            text=text_1,
            flores=flores_1,
            detector=detector,
            label=f"{label} - #1",
        )
        text_flores_2 = get_flores(
            text=text_2,
            flores=flores_2,
            detector=detector,
            label=f"{label} - #2",
        )

        if text_flores_1 != TARGET_LANG_FLORES or text_flores_2 != TARGET_LANG_FLORES:
            jsonData = {
                "input": {
                    "text_1": text_1,
                    "text_flores_1": text_flores_1,
                    "target_flores_1": TARGET_LANG_FLORES,
                    "target_score_max_1": TARGET_LANG_SCORE_MAX,
                    "detected_confidence_score_min_1": DETECTED_CONFIDENCE_SCORE_MIN,
                    "text_2": text_2,
                    "text_flores_2": text_flores_2,
                    "target_flores_2": TARGET_LANG_FLORES,
                    "target_score_max_2": TARGET_LANG_SCORE_MAX,
                    "detected_confidence_score_min_2": DETECTED_CONFIDENCE_SCORE_MIN,
                }
            }
            try:
                res = requests.post(
                    f"{translator_url}/predictions",
                    json=jsonData,
                    headers={"Content-Type": "application/json"},
                )
                if res.status_code != 200:
                    raise Exception(
                        f"Translation failed with status code: {res.status_code}"
                    )
                resJson = res.json()
                [translated_text_1, translated_text_2] = resJson["output"]
            except Exception as e:
                raise Exception(f"Translation failed with error: {e}")

            print(f'-- {label} - #1 - Original text is: "{text_1}" --')
            print(f'-- {label} - #1 - Translated text is: "{translated_text_1}" --')
            print(f'-- {label} - #2 - Original text is: "{text_2}" --')
            print(f'-- {label} - #2 - Translated text is: "{translated_text_2}" --')
        else:
            translated_text_1 = text_1
            translated_text_2 = text_2
            print(
                f"-- {label} - Texts are already in the correct language, no translation needed --"
            )
            print(f'-- {label} - #1 - Text is: "{translated_text_1}" --')
            print(f'-- {label} - #2 - Text is: "{translated_text_2}" --')

        endTimeTranslation = time.time()
        print(
            f"-- {label} - Translation completed in: {round((endTimeTranslation - startTimeTranslation) * 1000)} ms --"
        )

        return [translated_text_1, translated_text_2]


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
    text: str,
    flores: str,
    detector: any,
    label: str,
):
    if text == "":
        print(f"-- {label} - No text to give FLORES-200 for, skipping --")
        return TARGET_LANG_FLORES
    if flores is not None and flores != "":
        print(
            f'-- {label} - FLORES-200 code is given, skipping language auto-detection: "{flores}" --'
        )
        return flores

    text_flores = TARGET_LANG_FLORES
    confidence_values = detector.compute_language_confidence_values(text)
    target_lang_score = None
    detected_lang = None
    detected_lang_score = None

    print(f"-- Confidence values --")
    print(tabulate(confidence_values[:5]))

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
        and detected_lang_score > DETECTED_CONFIDENCE_SCORE_MIN
        and (target_lang_score is None or target_lang_score < TARGET_LANG_SCORE_MAX)
        and LANG_TO_FLORES.get(detected_lang.name) is not None
    ):
        text_flores = LANG_TO_FLORES[detected_lang.name]

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

    print(f'-- {label} - Selected text language FLORES-200: "{text_flores}" --')
    return text_flores
