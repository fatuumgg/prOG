## 1) Требования

- Python **3.10+**
- (опционально) Ollama для реальной LLM
- (опционально) `tiktoken` для точного токенайзинга
- (опционально) `sentence-transformers` для качественных эмбеддингов (RAG)
- (опционально) `pypdf` для PDF

---

## 2) Установка

### Вариант A: через venv (рекомендуется)

```bash
cd /path/to/projjjj
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

Установка зависимости:

Минимум (API + mock):
```bash
python -m pip install fastapi uvicorn pydantic requests pytest
```

Опционально для PDF:
```bash
python -m pip install pypdf
```

Опционально для точного токенайзинга:
```bash
python -m pip install tiktoken
```

Опционально для SBERT эмбеддингов (RAG):
```bash
python -m pip install sentence-transformers
```

---

## 3) Структура данных и файлов

Проект хранит данные на диске:

- **История диалогов**: `CE_DATA_STORE` (по умолчанию `./data`)
  - файл на диалог: `data/<conversation_id>.json`

- **RAG-векторное хранилище**: `CE_RAG_STORE` (по умолчанию `./rag_store.json`)
  - хранит пары `{chunk, vector}`

- **Память пользователя**: `CE_MEMORY_STORE` (по умолчанию `./user_memory.json`)
  - хранит факты по пользователям

---

## 4) Запуск (FastAPI)

### 4.1. Быстрый запуск в mock-режиме (без Ollama)

Подготовка:

```bash
cd /Users/.../projjjj
source .venv/bin/activate
```

Запуск сервера:

```bash
uvicorn chat_engine.app.api:app --reload --port 8000 --log-level info
```

Проверка:

```bash
curl -sS http://127.0.0.1:8000/docs >/dev/null && echo "API OK"
```

---

### 4.2. Запуск с Ollama (реальная LLM)

#### Шаг 1: запуск Ollama в отдельном терминале
(терминал №1)

```bash
ollama serve
```

Провека:

```bash
curl -sS http://127.0.0.1:11434/api/tags | python -m json.tool
```

#### Шаг 2: запуск API (терминал №2)

```bash
cd /Users/.../projjjj
source .venv/bin/activate

export CE_LLM=ollama
export CE_OLLAMA_URL="http://127.0.0.1:11434"
export CE_OLLAMA_MODEL="llama3.1:8b"

uvicorn chat_engine.app.api:app --reload --port 8000 --log-level debug
```

---

## 5) Настройки через переменные окружения

Главные переменные:

### Контекст / бюджет
- `CE_MAX_CONTEXT` — общий контекст (в токенах), напр. `2000`
- `CE_RESERVE_OUTPUT` — резерв под ответ модели, напр. `200`

### Токенайзер
- `CE_TOKENIZER=approx` (по умолчанию)
- `CE_TOKENIZER=tiktoken` (точнее)

### LLM
- `CE_LLM=mock` (по умолчанию)
- `CE_LLM=ollama`

### Summary
- `CE_ENABLE_SUMMARY=true/false`
- `CE_SUMMARY_MAX_TOKENS=256`
- `CE_SUMMARY_MIN_DROPPED=4`

### RAG
- `CE_ENABLE_RAG=true/false`
- `CE_RAG_MODE=off|auto|always`
  - `auto` — добавляет RAG только если запрос “похож на запрос по документам”
  - `always` — всегда пытается доставать фрагменты из базы документов
- `CE_RAG_TOPK=4`
- `CE_RAG_MAX_TOKENS=250` (лимит токенов на RAG-блок)
- `CE_EMBEDDER=hash|sbert`
- `CE_SBERT_MODEL="sentence-transformers/all-MiniLM-L6-v2"`

### Memory
- `CE_ENABLE_MEMORY=true/false`
- `CE_MEMORY_MAX_TOKENS=180`

### Пути хранения
- `CE_DATA_STORE=./data`
- `CE_RAG_STORE=./rag_store.json`
- `CE_MEMORY_STORE=./user_memory.json`

---

## 6) API эндпоинты

### `POST /chat`
Отправить сообщение в диалог.

Пример:
```bash
curl -sS -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"conversation_id":"demo1","user_id":"u_test","message":"Привет! Меня зовут Аня."}' \
| python -m json.tool
```

### `GET /conversations/{cid}`
Посмотреть историю диалога.

```bash
curl -sS http://127.0.0.1:8000/conversations/demo1 | python -m json.tool
```

### `GET /memory/{user_id}`
Посмотреть сохранённую память пользователя.

```bash
curl -sS http://127.0.0.1:8000/memory/u_test | python -m json.tool
```

### `DELETE /memory/{user_id}/{fact_id}`
Удалить конкретный факт.

```bash
curl -sS -X DELETE http://127.0.0.1:8000/memory/u_test/<fact_id> | python -m json.tool
```

### `POST /documents/upload`
Загрузить документ и проиндексировать в RAG.

Пример:
```bash
curl -sS -X POST "http://127.0.0.1:8000/documents/upload?user_id=u_test&replace=true" \
  -F "file=@./docs/book.pdf" \
| python -m json.tool
```

---

## 7) CLI режим

Можно общаться без FastAPI:

```bash
python -m chat_engine.app.cli --cid demo_cli --uid u_test
```

С отладкой:
```bash
python -m chat_engine.app.cli --cid demo_cli --uid u_test --debug
```

Индексировать документы через CLI:
```bash
python -m chat_engine.app.cli --cid demo_cli --uid u_test --ingest ./docs/book.pdf --ingest-replace
```

Команды в CLI:
- `/memory` — показать память
- `/forget <fact_id>` — удалить факт
- `/forget-key <key>` — удалить факты по ключу
- `/forget-all` — очистить память
- `/exit` — выход

---

## 8) Как запускать тесты

### Вариант 1: просто pytest (из корня проекта)

```bash
cd /Users/.../projjjj
source .venv/bin/activate
pytest -q
```

### Вариант 2: точечно один тест
```bash
pytest -q chat_engine/tests/test_rag.py
```

### Вариант 3: подробный вывод
```bash
pytest -vv
```

---

## 9) Пример полного запуска 

Терминал 1 (Ollama):
```bash
ollama serve
```

Терминал 2 (API):
```bash
cd /Users/.../projjjj
source .venv/bin/activate

export CE_LLM=ollama
export CE_TOKENIZER=tiktoken
export CE_EMBEDDER=sbert
export CE_SBERT_MODEL="sentence-transformers/all-MiniLM-L6-v2"

export CE_MAX_CONTEXT=2000
export CE_RAG_MAX_TOKENS=1200
export CE_RAG_MODE=auto

export CE_RAG_STORE="/Users/.../projjjj/rag_store.json"
export CE_MEMORY_STORE="/Users/.../projjjj/user_memory.json"
export CE_DATA_STORE="/Users/.../projjjj/data"

uvicorn chat_engine.app.api:app --reload --port 8000 --log-level debug
```

Терминал 3 (запрос):
```bash
curl -sS -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"conversation_id":"demo_auto2","user_id":"u_test","message":"По документам из ./docs/book.pdf: кто такой Томас Франк?"}' \
| python -m json.tool
```
