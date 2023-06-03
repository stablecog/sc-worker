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
modelsSpeakers = {"bark": []}
for language in bark_languages:
    for i in range(0, 10):
        modelsSpeakers["bark"].append(f"v2/{language}_speaker_{i}")
