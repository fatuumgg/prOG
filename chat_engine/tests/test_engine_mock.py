import tempfile

from chat_engine.adapters.repo_json import JsonFileConversationRepo
from chat_engine.adapters.llm_mock import EchoMockLLM
from chat_engine.adapters.tokens_approx import ApproxTokenCounter
from chat_engine.adapters.trunc_recency import RecencyTruncation
from chat_engine.use_cases.budget import Budget
from chat_engine.use_cases.chat_engine import ChatEngine

def test_engine_mock_works_and_saves_tokens_cache():
    with tempfile.TemporaryDirectory() as d:
        repo = JsonFileConversationRepo(d)
        engine = ChatEngine(
            repo=repo,
            llm=EchoMockLLM(),
            counter=ApproxTokenCounter(),
            budget=Budget(max_context_tokens=300, reserve_output_tokens=100),
            truncation=RecencyTruncation(),
            augmentors=[],
            system_prompt="SYS",
        )

        out = engine.handle_user_message("c1", "Привет")
        assert "[mock]" in out

        convo = repo.load("c1")
        assert convo.messages[0].role == "system"
        assert convo.messages[-2].role == "user"
        assert convo.messages[-1].role == "assistant"

        # проверяем кэш токенов
        assert isinstance(convo.messages[-1].meta.get("tokens"), int)
