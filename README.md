# GP-ITBot MVP RAG Telegram v2

Каркас MVP-прототипа Telegram-бота с RAG, OpenAI API, ChromaDB и расширенным логированием.

## Что изменено в v2

1. Расширено логирование:
   - `logs/runtime.log`;
   - таблица/БД `interactions`;
   - таблица/БД `runtime_events`;
   - логируются запуск, RAG-поиск, генерация ответа, ошибки, индексация документов и форма поддержки.

2. Системный промпт вынесен в отдельный файл:
   - `prompts/system_prompt.md`;
   - путь задается через `SYSTEM_PROMPT_FILE`.

3. Добавлено быстрое переключение логирования:
   - SQLite по умолчанию;
   - PostgreSQL через `LOG_BACKEND=postgres` и `docker-compose.postgres.yml`.

## Быстрый запуск с SQLite

```bash
cp .env.example .env
# заполнить TELEGRAM_BOT_TOKEN и OPENAI_API_KEY

docker compose up --build
```

## Индексация документов

Положите файлы `txt`, `md`, `docx`, `pdf` в папку:

```text
data/docs/
```

Запустите индексацию:

```bash
docker compose run --rm bot python ingest/ingest_docs.py
```

По умолчанию ingest пересоздаёт Chroma collection перед индексацией:

```env
RECREATE_CHROMA_COLLECTION=true
```

Это нужно, чтобы после изменения документов или стратегии чанкинга в базе не оставались старые чанки.

Если нужен инкрементальный режим, можно поставить:

```env
RECREATE_CHROMA_COLLECTION=false
```

Проверить состояние коллекции после ingest:

```bash
python scripts/check_chroma_collection.py
```

## Запуск с PostgreSQL для логирования

В `.env` установите:

```env
LOG_BACKEND=postgres
POSTGRES_DB=gp_itbot_logs
POSTGRES_USER=gp_itbot
POSTGRES_PASSWORD=gp_itbot_password
```

Запуск:

```bash
docker compose -f docker-compose.postgres.yml up --build
```

## Где редактировать системный промпт

```text
prompts/system_prompt.md
```

После изменения промпта перезапустите контейнер бота.

## Команды бота

- `/start` — приветствие;
- `/support` — форма передачи обращения в поддержку;
- любой текст — вопрос к RAG-ассистенту.

## Логирование

### Runtime-лог

```text
logs/runtime.log
```

### SQLite

```text
logs/app_logs.sqlite3
```

### PostgreSQL

Таблицы:

```sql
interactions
runtime_events
```

## Ограничения MVP

- Нет интеграции с реальной helpdesk-системой.
- Нет персонального контекста пользователя.
- Бот отвечает только по общей базе знаний.
- Секретные данные маскируются перед сохранением в логи.
