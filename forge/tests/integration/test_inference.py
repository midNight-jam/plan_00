from openai import OpenAI

# 1. Point the official OpenAI client to your local Forge server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="forge-local-key" # We haven't implemented auth yet, so any string works!
)

def test_forge_stream():
    print("Initiating connection to Forge...\n")
    print("-" * 50)
    
    # 2. Call your endpoint exactly as if it were ChatGPT
    stream = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        messages=[
            {"role": "system", "content": "You are Forge, a bare-metal AI orchestrator."},
            {"role": "user", "content": "Give me a brief, one-paragraph status report on your operational readiness running on an RTX 5080."}
        ],
        stream=True,
        temperature=0.7,
        max_tokens=150
    )

    # 3. Iterate over the SSE chunks and print smoothly
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content is not None:
            # Print without newlines and flush the buffer immediately
            print(content, end="", flush=True)
            
    print("\n" + "-" * 50)
    print("[Stream Complete]")

if __name__ == "__main__":
    test_forge_stream()