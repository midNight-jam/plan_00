import time
import uuid
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    role:str
    content:str


"""
Standardization by adhering to this structure, any existing tool that knows how to talk to
OpenAI will know how to talk to our Forge server.
"""
class ChatCompletionRequest(BaseModel):
    model:str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 8192
    stream: Optional[bool] = False

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory = lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Optional[Dict[str, int]] = None

class HealthResponse(BaseModel):
    status: str
    model: str
    version: str = "0.1.0"

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = 1686935002
    owned_by: str = "forge"

class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelCard]