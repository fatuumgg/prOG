import tempfile

from chat_engine.adapters.repo_json import JsonFileConversationRepo
from chat_engine.adapters.llm_mock import EchoMockLLM
from chat_engine.adapters.tokens_approx import ApproxTokenCounter
from chat_engine.adapters.trunc_recency import RecencyTruncation
from chat_engine.adapters.summarizer_mock import MockSummarizer
from chat_engine.use_cases.budget import Budget
from chat_engine.use_cases.chat_engine import ChatEngine
from chat_engine.use_cases.summary_buffer import SummaryPolicy

def test_summary_reduces_tokens_and_context_fits():
    with tempfile.TemporaryDirectory() as d:
        repo = JsonFileConversationRepo(d)
        counter = ApproxTokenCounter()
        trunc = RecencyTruncation()

        engine = ChatEngine(
            repo=repo,
            llm=EchoMockLLM(),
            counter=counter,
            budget=Budget(max_context_tokens=220, reserve_output_tokens=60, safety_margin_tokens=0),
            truncation=trunc,
            augmentors=[],
            system_prompt="SYS",
            summarizer=MockSummarizer(),
            summary_policy=SummaryPolicy(min_dropped_messages=1, max_summary_tokens=80),
        )

        for i in range(12):
            engine.handle_user_message("c1", "hello " + ("x" * 80) + f" #{i}")

        convo = repo.load("c1")

        summaries = [m for m in convo.messages if m.role == "system" and m.meta.get("type") == "summary"]
        assert len(summaries) >= 1
        sm = summaries[0]

        replaced = sm.meta.get("replaced", {})
        assert isinstance(replaced.get("tokens"), int)
        assert isinstance(sm.meta.get("tokens"), int)
        assert sm.meta["tokens"] < replaced["tokens"]

        context = list(convo.messages)
        fitted = trunc.fit(context, counter=counter, max_input_tokens=engine.budget.max_input_tokens)
        assert counter.count_messages(fitted) <= engine.budget.max_input_tokens

        assert convo.messages[-2].role == "user"
        assert convo.messages[-1].role == "assistant"
