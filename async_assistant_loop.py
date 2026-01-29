"""
async_assistant_loop.py
Fully‑async REPL that delegates all heavy lifting to core_assistant.run_assistant_async.

Thread-based context management:
- OpenAI threads store full conversation history server-side via thread_id
- Optimized for GPT-4.1 with 1M token context window support
"""

import asyncio, openai, os
from core_assistant import run_assistant_async, create_context_prompt_with_budget

ASSISTANT_ID = os.getenv("ASSISTANT_ID")           # export beforehand
MODEL = "gpt-4.1"  # GPT-4.1 with 1M context window

# GPT-4.1 optimized token budget (increased for richer context)
DEFAULT_TOKEN_BUDGET = 96_000  # Increased from 64k, thread manages full history

async def session():
    thread = openai.beta.threads.create()
    print("🩺  Medical‑Affairs async REPL (GPT-4.1) – type 'exit' to quit\n")

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
            token_budget=DEFAULT_TOKEN_BUDGET,  # GPT-4.1 optimized
            has_files=False,
        )

        reply = await run_assistant_async(
            thread_id=thread.id,
            assistant_id=ASSISTANT_ID,
            prompt=prompt,
        )
        print(f"\nAssistant> {reply}\n")

if __name__ == "__main__":
    asyncio.run(session())
