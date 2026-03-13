# Smart Billboard Geo Search (Weighted RRF Orchestrator)

Мультимодальный REST API: image + audio + text + coords -> Top-5 рекламных поверхностей.

## Что сделано под роль «Леха» (Infra / Prompt Engineer)

- **Docker Compose** с healthchecks, `depends_on: service_healthy`, общими volume для CSV/индекса.
- **Regex & Prompts**:
  - строгий GID-паттерн `^[A-Z]{4}\d{5}[A-ZА-Я]$`
  - нормализация OCR-путаницы `A/А`, `B/В`
  - `INITIAL_WHISPER_PROMPT` с доменными терминами: *Ангарск, билборд, широта, долгота*.
- **NLP extraction**:
  - Natasha (`AddrExtractor`) для извлечения адресных сущностей из текста/ASR
  - fallback regex для адресов, если Natasha недоступна.
- **Mock DB**:
  - отдача структурированных JSON-карточек по GID (`/objects/{gid}`)
  - список `/objects`, поиск и `/nearest`.
- **Тестирование модальностей**:
  - `scripts/test_requests.py` покрывает text+coords, audio, image, full multimodal.

## Общий pipeline (Вариант №3)

1. Оркестратор принимает multipart (`/search`) и параллельно запускает сервисы (`asyncio.gather`).
2. OCR/ASR + NLP извлекают GID, координаты, адреса.
3. Retrieval:
   - visual Top-10
   - semantic Top-10
   - lexical Top-10 (RapidFuzz)
   - geo candidates (до 1 км)
4. Fusion через Weighted RRF:
   - exact GID: OCR=1.0 (short-circuit)
   - multimodal: visual=0.4, semantic=0.3, lexical=0.15, geo=0.15
5. Финализация: confidence (sigmoid), enrich карточками из mock DB.

## Быстрый запуск

```bash
docker compose up --build
```

Swagger: `http://localhost:8000/docs`

## Локальные проверки

```bash
pip install -r requirements-dev.txt
pytest -q
python scripts/index_chroma.py
python scripts/test_requests.py
```
