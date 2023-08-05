from lingua import Language
from dotenv import load_dotenv
import os

load_dotenv()

TRANSLATOR_MODEL_ID = "facebook/nllb-200-distilled-1.3B"
TRANSLATOR_CACHE = "translator-cache"

LANG_TO_FLORES = {
    "AFRIKAANS": "afr_Latn",
    "ALBANIAN": "als_Latn",
    "ARABIC": "arb_Arab",
    "ARMENIAN": "hye_Armn",
    "AZERBAIJANI": "azj_Latn",
    "BASQUE": "eus_Latn",
    "BELARUSIAN": "bel_Cyrl",
    "BENGALI": "ben_Beng",
    "BOKMAL": "nob_Latn",
    "BOSNIAN": "bos_Latn",
    "BULGARIAN": "bul_Cyrl",
    "CATALAN": "cat_Latn",
    "CHINESE": "zho_Hans",
    "CROATIAN": "hrv_Latn",
    "CZECH": "ces_Latn",
    "DANISH": "dan_Latn",
    "DUTCH": "nld_Latn",
    "ENGLISH": "eng_Latn",
    "ESPERANTO": "epo_Latn",
    "ESTONIAN": "est_Latn",
    "FINNISH": "fin_Latn",
    "FRENCH": "fra_Latn",
    "GANDA": "lug_Latn",
    "GEORGIAN": "kat_Geor",
    "GERMAN": "deu_Latn",
    "GREEK": "ell_Grek",
    "GUJARATI": "guj_Gujr",
    "HEBREW": "heb_Hebr",
    "HINDI": "hin_Deva",
    "HUNGARIAN": "hun_Latn",
    "ICELANDIC": "isl_Latn",
    "INDONESIAN": "ind_Latn",
    "IRISH": "gle_Latn",
    "ITALIAN": "ita_Latn",
    "JAPANESE": "jpn_Jpan",
    "KAZAKH": "kaz_Cyrl",
    "KOREAN": "kor_Hang",
    "LATVIAN": "lvs_Latn",
    "LITHUANIAN": "lit_Latn",
    "MACEDONIAN": "mkd_Cyrl",
    "MALAY": "zsm_Latn",
    "MAORI": "mri_Latn",
    "MARATHI": "mar_Deva",
    "MONGOLIAN": "khk_Cyrl",
    "NYNORSK": "nno_Latn",
    "PERSIAN": "pes_Arab",
    "POLISH": "pol_Latn",
    "PORTUGUESE": "por_Latn",
    "PUNJABI": "pan_Guru",
    "ROMANIAN": "ron_Latn",
    "RUSSIAN": "rus_Cyrl",
    "SERBIAN": "srp_Cyrl",
    "SHONA": "sna_Latn",
    "SLOVAK": "slk_Latn",
    "SLOVENE": "slv_Latn",
    "SOMALI": "som_Latn",
    "SOTHO": "nso_Latn",
    "SPANISH": "spa_Latn",
    "SWAHILI": "swh_Latn",
    "SWEDISH": "swe_Latn",
    "TAGALOG": "tgl_Latn",
    "TAMIL": "tam_Taml",
    "TELUGU": "tel_Telu",
    "THAI": "tha_Thai",
    "TSONGA": "tso_Latn",
    "TURKISH": "tur_Latn",
    "UKRAINIAN": "ukr_Cyrl",
    "URDU": "urd_Arab",
    "VIETNAMESE": "vie_Latn",
    "XHOSA": "xho_Latn",
    "YORUBA": "yor_Latn",
    "ZULU": "zul_Latn",
}

FLORES_TO_LANG = dict([(value, key) for key, value in LANG_TO_FLORES.items()])

TARGET_LANG = Language.ENGLISH
TARGET_LANG_FLORES = LANG_TO_FLORES[TARGET_LANG.name]
TARGET_LANG_SCORE_MAX = 0.88
DETECTED_CONFIDENCE_SCORE_MIN = 0.1
TRANSLATOR_COG_URL = os.environ.get("TRANSLATOR_COG_URL", None)
