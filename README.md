# Weighted RRF Orchestrator (Вариант №3)

Асинхронный API, который объединяет OCR/ASR/Visual/Text/Geo сигналы и отдает Top-5 билбордов через `Weighted RRF`.

## Что реализовано

- `FastAPI` endpoint `POST /search` (`multipart/form-data`): image, audio, text, latitude, longitude.
- Асинхронный оркестратор на `asyncio.gather`:
  - OCR (PaddleOCR placeholder + fallback)
  - ASR (faster-whisper + initial prompt)
  - Visual embedding (OpenCLIP placeholder + deterministic fallback)
- NLP extraction:
  - Regex-валидатор GID: `^[A-Z]{4}\d{5}[А-ЯA-Z]$`
  - Natasha для адресных сущностей
- Retrieval каналы:
  - Visual top-k
  - Semantic top-k
  - Lexical top-k (RapidFuzz)
  - Geo filter (geopy)
- Fusion:
  - Priority Case A (точный GID → 100%)
  - Case B weighted RRF
  - Platt scaling в вероятность 0–100
- Mock DB по CSV (полные карточки по `gid`)
- Docker + Docker Compose (API + ChromaDB + volume для model cache)

## Запуск локально

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Запуск через Docker

```bash
docker compose up --build
```

## Подготовка индекса

```bash
python app/scripts/build_index.py --input data/billboards_raw.csv --output data/billboards.csv
```

## Формула

\[
Score(d) = \sum_r w_r \cdot \frac{1}{k + rank_r(d)}
\]

Потом:

\[
P(y=1|d)=\sigma(a\cdot Score(d)+b)
\]
