import pytest
from httpx import AsyncClient, ASGITransport
from src.forge.server import app

@pytest.mark.asyncio
async def test_health_check():
    """Verify the /health endpoint returns healthy."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

@pytest.mark.asyncio
async def test_list_models():
    """Verify the /v1/models endpoint returns the loaded model."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert len(data["data"]) > 0

@pytest.mark.asyncio
async def test_chat_completion_non_streaming():
    """Verify non-streaming chat completions return a valid response."""
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "messages": [
            {"role": "user", "content": "Hello! Give me a one word response."}
        ],
        "max_tokens": 10,
        "stream": false
    }
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/v1/chat/completions", json=payload)
        
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    content = data["choices"][0]["message"]["content"]
    assert len(content) > 0