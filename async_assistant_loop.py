"""
async_assistant_loop.py
Fully-async REPL that delegates all heavy lifting to core_assistant.run_responses_async.

v5.0 – Responses API Migration:
- No threads – conversation continuity via previous_response_id
- Uses run_responses_async for async execution
"""

import asyncio, os
from core_assistant import run_responses_async, create_context_prompt_with_budget

MODEL = "gpt-4.1"
DEFAULT_TOKEN_BUDGET = 96_000


async def session():
    print("Medical-Affairs async REPL (GPT-4.1 Responses API) - type 'exit' to quit\n")

    previous_response_id = None

    while True:
        user_input = input("You> ").strip()
        if not user_input or user_input.lower() == "exit":
            break

        prompt = create_context_prompt_with_budget(
            user_input,
            output_type="detailed_analysis",
            response_tone="professional",
            compliance_level="strict",
            role="",
            client="",
            history=[],
            token_budget=DEFAULT_TOKEN_BUDGET,
            has_files=False,
        )

        reply, response_id, tool_log = await run_responses_async(
            model=MODEL,
            input_text=prompt,
            previous_response_id=previous_response_id,
        )
        previous_response_id = response_id
        print(f"\nAssistant> {reply}\n")

if __name__ == "__main__":
    asyncio.run(session())
