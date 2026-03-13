# Smart Billboard Geo Search (Hackathon Case #2)

Мультимодальный REST API для поиска рекламных поверхностей по изображению, тексту, аудио и координатам.

## Архитектура (вариант 3)

- **Orchestrator/Gateway (FastAPI)**: принимает `image`, `audio`, `text`, `lat/lon`, проводит инвентаризацию модальностей и параллельно запускает сервисы через `asyncio.gather`.
- **OCR & ASR service**:
  - Regex GID: `^[A-ZА-Я]{4}\d{5}[A-ZА-Я]$`
  - initial prompt для Whisper-подобного пайплайна вынесен в константу `INITIAL_WHISPER_PROMPT`.
- **Vector search service**: mock-векторизация и Top-K поиск (структурно совместим с идеей OpenCLIP + ChromaDB).
- **Geo service**: фильтрация по радиусу 100–200 м (haversine), ранжирование по близости.
- **Mock DB service**: читает CSV и возвращает структурированные карточки объектов по GID.
- **Final Fusion**: Weighted RRF с динамическими весами.

## Репозиторий

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

## Быстрый запуск

```bash
docker compose up --build
```

Orchestrator доступен на `http://localhost:8000`.

- Swagger: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

## Примеры запросов

### 1) Только текст + координаты

```bash
curl -X POST http://localhost:8000/search \
  -F 'text=щит рядом с Тверской' \
  -F 'lat=55.757' \
  -F 'lon=37.615'
```

### 2) Аудио (как текстовый mock payload)

```bash
curl -X POST http://localhost:8000/search \
  -F 'audio=@sample.txt'
```

## Индексация (первые 4 часа)

```bash
python scripts/index_chroma.py
```

Скрипт создает mock-индекс `data/mock_index.json` из CSV.

## Тесты

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Что покрыто под роль Infrastructure & Prompt Engineer

- Docker Compose для всех микросервисов.
- Regex-паттерн GID и выделение координат из текста/ASR.
- initial_prompt для ASR-транскрипции рекламных терминов.
- Mock database слой из CSV в JSON-карточки.
- Набор тестовых запросов (`scripts/test_requests.py`) и unit-тесты.
