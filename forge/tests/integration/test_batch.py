import asyncio
import time
from openai import AsyncOpenAI

# 1. Use the Async version of the OpenAI client
client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="forge-local-key"
)

async def fetch_response(worker_id: int, prompt: str):
    """Simulates a single user sending a request."""
    print(f"[User {worker_id}] Sending prompt: '{prompt[:35]}...'")
    start_time = time.time()
    
    # We disable streaming here just so our terminal output doesn't turn into a jumbled mess
    # when 5 streams try to print letters to the console at the exact same time.
    response = await client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        messages=[
            {"role": "system", "content": "You are a concise AI."},
            {"role": "user", "content": prompt}
        ],
        stream=False,
        temperature=0.7,
        max_tokens=100
    )
    
    end_time = time.time()
    duration = end_time - start_time
    text = response.choices[0].message.content.strip().replace('\n', ' ')
    
    print(f"[User {worker_id}] Finished in {duration:.2f}s | Reply: '{text[:150]}...'")

async def main():
    # A list of completely different prompts of varying lengths
    prompts = [
        "Explain the theory of relativity in one sentence.",
        "Write a haiku about a powerful NVIDIA GPU.",
        "What is the capital of France?",
        "Write a quick Python function to reverse a string.",
        "Who wrote the sci-fi book 'Dune'?"
    ]
    
    print(f"Firing {len(prompts)} concurrent requests to Forge...\n")
    print("-" * 60)
    
    start_time = time.time()
    
    # 2. Fire all requests at the exact same time using asyncio.gather
    tasks = [fetch_response(i, prompt) for i, prompt in enumerate(prompts)]
    await asyncio.gather(*tasks)
    
    total_duration = time.time() - start_time
    
    print("-" * 60)
    print(f"All {len(prompts)} requests completed in {total_duration:.2f} seconds.")

if __name__ == "__main__":
    asyncio.run(main())