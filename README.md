# Smart Billboard Geo Search (Hackathon Case #2)

Мультимодальный REST API для поиска рекламных поверхностей по изображению, тексту, аудио и координатам.

## Архитектура (вариант 3)

- **Orchestrator/Gateway (FastAPI)**: принимает `image`, `audio`, `text`, `lat/lon`, делает инвентаризацию модальностей и параллельно запускает сервисы через `asyncio.gather`.
- **OCR & ASR service**:
  - строгий GID regex: `^[A-Z]{4}\d{5}[A-ZА-Я]$`
  - нормализация OCR-путаницы кириллица/латиница в префиксе (например `В` → `B`)
  - выделение координат с поддержкой `.`/`,` и валидацией диапазонов lat/lon
  - `INITIAL_WHISPER_PROMPT` с рекламными терминами для повышения качества транскрипции.
- **Vector search service**: mock-векторизация и Top-K поиск (структурно совместим с OpenCLIP + ChromaDB).
- **Geo service**: фильтрация по радиусу 100–200 м (haversine), ранжирование по близости.
- **Mock DB service**: читает CSV и возвращает структурированные карточки (вложенные coordinates + тип поверхности).
- **Final Fusion**: Weighted RRF с динамическими весами.

## Repository layout

```text
services/
  orchestrator/
  ocr_asr/
  vector_search/
  geo/
  mock_db/
data/billboards.csv
scripts/index_chroma.py
scripts/test_requests.py
docker-compose.yml
```

## Infrastructure focus (Docker)

- единое окружение через `docker-compose.yml`
- healthcheck для каждого сервиса
- `depends_on: condition: service_healthy` для orchestrator
- оптимизация Docker-слоев: отдельный слой для `requirements.txt`, затем копирование кода
- `.dockerignore` для уменьшения контекста сборки.

## Быстрый запуск

```bash
docker compose up --build
```

Orchestrator: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

## Примеры ручных проверок (curl)

### text + coords

```bash
curl -X POST http://localhost:8000/search \
  -F 'text=щит рядом с Тверской' \
  -F 'lat=55.757' \
  -F 'lon=37.615'
```

### audio

```bash
curl -X POST http://localhost:8000/search \
  -F 'audio=@sample.txt'
```

### image

```bash
curl -X POST http://localhost:8000/search \
  -F 'image=@sample.jpg'
```

## Индексация (первые 4 часа)

```bash
python scripts/index_chroma.py
```

Скрипт создает mock-индекс `data/mock_index.json` из CSV.

## Тестирование

```bash
pip install -r requirements-dev.txt
pytest -q
python scripts/test_requests.py
```

