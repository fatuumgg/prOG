# Chat Engine — управление контекстом диалога (truncation + summary + RAG + user memory)

Проект реализует систему управления контекстом диалога с LLM:
- хранит историю сообщений (user/assistant/system) на диске,
- считает токены и **обрезает контекст** при превышении лимита,
- умеет **суммаризировать** “старую” историю (summary buffer),
- умеет **RAG**: индексировать документы (TXT/PDF) и добавлять релевантные фрагменты в контекст,
- умеет **персистентную память о пользователе** (извлечение фактов → хранение → подмешивание в новые диалоги),
- поддерживает **mock-режим** без реальной LLM.

---

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

Установи зависимости:

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

Открой терминал:

```bash
cd /Users/fatuum/Desktop/projjjj
source .venv/bin/activate
```

Запусти сервер:

```bash
uvicorn chat_engine.app.api:app --reload --port 8000 --log-level info
```

Проверка:

```bash
curl -sS http://127.0.0.1:8000/docs >/dev/null && echo "API OK"
```

---

### 4.2. Запуск с Ollama (реальная LLM)

#### Шаг 1: запусти Ollama в отдельном терминале
(терминал №1)

```bash
ollama serve
```

Проверь что Ollama отвечает:

```bash
curl -sS http://127.0.0.1:11434/api/tags | python -m json.tool
```

#### Шаг 2: запусти API (терминал №2)

```bash
cd /Users/fatuum/Desktop/projjjj
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

## 6) Как это работает (принцип работы)

### 6.1. История диалога
Каждый запрос:
1) Загружает диалог из JSON (`JsonFileConversationRepo`)
2) Убеждается, что есть system-сообщение (system prompt)
3) Добавляет новое сообщение пользователя
4) Сохраняет новые сообщения (после ответа)

### 6.2. Token counting
`TokenCounter` считает токены:
- `ApproxTokenCounter` — грубо (≈ 1 токен на 4 символа)
- `TiktokenTokenCounter` — точнее

Токены кэшируются в `message.meta["tokens"]`.

### 6.3. Обрезание (Truncation)
`RecencyTruncation`:
- **всегда сохраняет pinned-сообщения** (`meta.pinned=True`) — например system prompt, summary, память пользователя.
- остальные сообщения добавляет **с конца (самые свежие)**, пока не превысим бюджет.

### 6.4. Summary buffer (суммаризация)
Если включено summary:
- когда truncation “выкидывает” слишком много сообщений,
- система создаёт **одно system summary-сообщение**, которое заменяет старую часть истории
- summary кладётся в начало (после system prompt), pinned=True
- при следующих суммаризациях summary “роллится” (обновляется)

### 6.5. RAG (документы)
RAG состоит из:
- загрузчика (TXT/PDF),
- чанкер (нарезка на куски),
- эмбеддер (hash или sbert),
- vector store (JSON).

Индексация:
1) `RagIndexer` → loader → chunker → embedder → store.upsert()

При чате:
- `RagAugmentor` может вставить system-блок retrieved_context
- режим:
  - `auto`: вставляет только если запрос выглядит как “в документе/по документу/pdf…”
  - `always`: всегда пытается вставить retrieved_context

### 6.6. Память пользователя (как ChatGPT Memory)
Пайплайн:
1) `RuleBasedMemoryExtractor` вынимает факты из user-сообщения (имя, нравится/не нравится, цель, проект…)
2) `MemoryManager` делает upsert в `JsonUserMemoryStore` (стабильный fact_id по user_id+key)
3) `MemoryAugmentor` добавляет system-блок “Профиль пользователя” в контекст

---

## 7) API эндпоинты

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

## 8) CLI режим

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

## 9) Как запускать тесты

### Вариант 1: просто pytest (из корня проекта)

```bash
cd /Users/fatuum/Desktop/projjjj
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

## 10) Типичные проблемы

### 1) `Connection refused 127.0.0.1:11434`
Значит Ollama не запущена. Запусти:
```bash
ollama serve
```

### 2) RAG не вставляется в режиме auto
В `auto` RAG добавляется только если запрос выглядит как запрос “по документам”.
Либо:
- пиши запрос типа: `По документу ...`, `в PDF ...`, `в книге ...`
- либо включи:
```bash
export CE_RAG_MODE=always
```

### 3) Тест `test_rag` падает
Если augmentor в `auto`, а запрос не “про документы” — RAG не вставится.
В тесте нужно `mode="always"` (или другой текст запроса).

---

## 11) Пример полного запуска (как у тебя)

Терминал 1 (Ollama):
```bash
ollama serve
```

Терминал 2 (API):
```bash
cd /Users/fatuum/Desktop/projjjj
source .venv/bin/activate

export CE_LLM=ollama
export CE_TOKENIZER=tiktoken
export CE_EMBEDDER=sbert
export CE_SBERT_MODEL="sentence-transformers/all-MiniLM-L6-v2"

export CE_MAX_CONTEXT=2000
export CE_RAG_MAX_TOKENS=1200
export CE_RAG_MODE=auto

export CE_RAG_STORE="/Users/fatuum/Desktop/projjjj/rag_store.json"
export CE_MEMORY_STORE="/Users/fatuum/Desktop/projjjj/user_memory.json"
export CE_DATA_STORE="/Users/fatuum/Desktop/projjjj/data"

uvicorn chat_engine.app.api:app --reload --port 8000 --log-level debug
```

Терминал 3 (запрос):
```bash
curl -sS -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"conversation_id":"demo_auto2","user_id":"u_test","message":"По документам из ./docs/book.pdf: кто такой Томас Франк?"}' \
| python -m json.tool
```
