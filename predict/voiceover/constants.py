models = ["bark"]
bark_languages = [
    "en",
    "zh",
    "fr",
    "de",
    "hi",
    "it",
    "ja",
    "ko",
    "pl",
    "pt",
    "ru",
    "es",
    "tr",
]
models_speakers = {"bark": []}
for language in bark_languages:
    for i in range(0, 10):
        models_speakers["bark"].append(f"v2/{language}_speaker_{i}")
for i in range(0, 18):
    models_speakers["bark"].append(f"c_en_{i}")
