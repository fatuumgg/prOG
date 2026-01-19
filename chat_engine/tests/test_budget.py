from chat_engine.use_cases.budget import Budget

def test_budget_max_input_tokens():
    b = Budget(max_context_tokens=1000, reserve_output_tokens=200, safety_margin_tokens=50)
    assert b.max_input_tokens == 750
