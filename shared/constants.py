APP_VERSION = "0.1.0"

SERVICE_NAME_ORCHESTRATOR = "orchestrator"
SERVICE_NAME_OCR_ASR = "ocr_asr"
SERVICE_NAME_VECTOR_SEARCH = "vector_search"
SERVICE_NAME_GEO = "geo"
SERVICE_NAME_MOCK_DB = "mock_db"

DEFAULT_TOP_K = 20
MAX_TOP_K = 100
TOP5_LIMIT = 5

DEFAULT_TIMEOUT_SECONDS = 5.0

SOURCE_WEIGHTS = {
    "vector": 1.00,
    "geo": 0.85,
    "ocr": 0.95,
    "asr": 0.90,
}