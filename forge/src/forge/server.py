import os
from fastapi import FastAPI, HTTPException
from .models import (
    ChatCompletionRequest, 
    ChatCompletionResponse, 
    ChatCompletionResponseChoice,
    ChatMessage
)
from .engine import ForgeEngine

app = FastAPI(title="zzForge Inference Server")

# INITIALIZATION :  instantiates the engine globally so it loads the model weights 
# into VRAM once upon server startup, rather than on every request.
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
engine = ForgeEngine(model_name=MODEL_NAME)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    try:
        final_text = ""

        # CONTINUOS BATCHING: We await the generator.
        # vLLM yields the full accumulated text on every single iteration.
        # For this Standard (non-streaming) endpoint, we overwrite the variable
        # untill the loop finishes, capturing only the final, complete string.
        async for text in engine.generate(request):
            final_text = text

        # Formatting: Pack the raw string back into the strict OpenAI Pydantic schema
        choice = ChatCompletionResponseChoice(
            index = 0,
            message=ChatMessage(role="assistant", content=final_text),
            finish_reason="stop"
        )

        return ChatCompletionResponse(
            model=request.model,
            choices=[choice]
        )
    
    except Exception as e:
        # SAFETY: Catch vLLM crashes (like OOM errors) and surface them cleanlu
        raise HTTPException(status_code=500, detail=str(e))
