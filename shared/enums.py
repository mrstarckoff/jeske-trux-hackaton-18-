from enum import Enum


class SourceType(str, Enum):
    OCR = "ocr"
    ASR = "asr"
    VECTOR = "vector"
    GEO = "geo"
    MOCK_DB = "mock_db"
    ORCHESTRATOR = "orchestrator"


class ModalityType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    COORDS = "coords"