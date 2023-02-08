from lingua import Language
import time
from .constants import LANG_TO_FLORES
import requests

target_lang_score_max = 0.9
target_lang = Language.ENGLISH
target_lang_flores = LANG_TO_FLORES[target_lang.name]


def translate_text(text, flores_200_code, text_2, flores_200_code_2, translator_url, detector, label):
    print(f"-- {label} - Translator url is: '{translator_url}' --")

    if text == "" and text_2 == "":
        print(f"-- {label} - No text to translate, skipping --")
        return ["", ""]

    startTimeTranslation = time.time()

    translated_text = ""
    translated_text_2 = ""

    text_lang_flores = target_lang_flores
    text_lang_flores_2 = target_lang_flores

    text_lang_flores = get_flores_200_code(
        text, flores_200_code, target_lang_flores, detector, f"{label} - #1")
    text_lang_flores_2 = get_flores_200_code(
        text_2, flores_200_code_2, target_lang_flores, detector, f"{label} - #2")

    if text_lang_flores != target_lang_flores or text_lang_flores_2 != target_lang_flores:
        jsonData = {
            "input": {
                "text": text,
                "text_lang": text_lang_flores,
                "target_lang": target_lang_flores,
                "text_2": text_2,
                "text_lang_2": text_lang_flores_2,
                "target_lang_2": target_lang_flores,
            }
        }
        res = requests.post(
            f"{translator_url}/predictions",
            json=jsonData,
            headers={'Content-Type': 'application/json'}
        )
        if res.status_code != 200:
            raise Exception(
                f"Translation failed with status code: {res.status_code}")
        resJson = res.json()
        [translated_text, translated_text_2] = resJson["output"]
        print(f'-- {label} - #1 - Original text is: "{text}" --')
        print(f'-- {label} - #1 - Translated text is: "{translated_text}" --')
        print(f'-- {label} - #2 - Original text is: "{text_2}" --')
        print(f'-- {label} - #2 - Translated text is: "{translated_text_2}" --')
    else:
        translated_text = text
        translated_text_2 = text_2
        print(
            f"-- {label} - Texts are already in the correct language, no translation needed --"
        )
        print(f'-- {label} - #1 - Text is: "{translated_text}" --')
        print(f'-- {label} - #2 - Text is: "{translated_text_2}" --')

    endTimeTranslation = time.time()
    print(f"-- {label} - Translation completed in: {round((endTimeTranslation - startTimeTranslation) * 1000)} ms --")

    return [translated_text, translated_text_2]


def get_flores_200_code(text, defined_flores_code, target_lang_flores, detector, label):
    if text == "":
        return target_lang_flores
    if defined_flores_code is not None:
        print(
            f'-- {label} - FLORES-200 code is given, skipping language auto-detection: "{defined_flores_code}" --')
        return defined_flores_code

    text_lang_flores = target_lang_flores
    confidence_values = detector.compute_language_confidence_values(text)
    target_lang_score = None
    detected_lang = None
    detected_lang_score = None

    print(f'-- Confidence values - {confidence_values[:10]} --')
    for index in range(len(confidence_values)):
        curr = confidence_values[index]
        if index == 0:
            detected_lang = curr[0]
            detected_lang_score = curr[1]
        if curr[0] == Language.ENGLISH:
            target_lang_score = curr[1]

    if detected_lang is not None and detected_lang != target_lang and (target_lang_score is None or target_lang_score < target_lang_score_max) and LANG_TO_FLORES.get(detected_lang.name) is not None:
        text_lang_flores = LANG_TO_FLORES[detected_lang.name]

    if detected_lang is not None:
        print(
            f'-- {label} - Guessed text language: "{detected_lang.name}". Score: {detected_lang_score} --')
    if detected_lang is not None and target_lang_score is not None and detected_lang != target_lang:
        print(f'-- {label} - Target language score: {target_lang_score} --')

    print(f'-- {label} - Selected text language FLORES-200: "{text_lang_flores}" --')
    return text_lang_flores
