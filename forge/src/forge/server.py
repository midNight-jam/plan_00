import os
from fastapi import FastAPI, HTTPException
from .models import (
    ChatCompletionRequest, 
    ChatCompletionResponse, 
    ChatCompletionResponseChoice,
    ChatMessage
)
import uuid
from fastapi import Request, HTTPException
from vllm import SamplingParams
from .engine import ForgeEngine

app = FastAPI(title="zzForge Inference Server")

# INITIALIZATION :  instantiates the engine globally so it loads the model weights 
# into VRAM once upon server startup, rather than on every request.
# MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
# MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct") # this failed to loadin gpu :(, thus going for AWQ
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen2.5-7B-Instruct-AWQ")
engine = ForgeEngine(model_name=MODEL_NAME)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest): # Using Pydantic model
    try:

        #. Send the validated Pydantic model to the vLLM engine
        results_generator = engine.generate(request)
        
        # 6. Await the final result (for non-streaming)
        generated_text = ""
        final_output = None
        async for text in results_generator:
            generated_text = text # engine.py yields the text string directly
            
        
        return {
            "id": f"chat-{uuid.uuid4()}",
            "object": "chat.completion",
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": "stop"
            }]
        }

        # # CONTINUOS BATCHING: We await the generator.
        # # vLLM yields the full accumulated text on every single iteration.
        # # For this Standard (non-streaming) endpoint, we overwrite the variable
        # # untill the loop finishes, capturing only the final, complete string.
        # async for text in engine.generate(request):
        #     final_text = text

        # # Formatting: Pack the raw string back into the strict OpenAI Pydantic schema
        # choice = ChatCompletionResponseChoice(
        #     index = 0,
        #     message=ChatMessage(role="assistant", content=final_text),
        #     finish_reason="stop"
        # )

        # return ChatCompletionResponse(
        #     model=request.model,
        #     choices=[choice]
        # )
    
    except Exception as e:
        # SAFETY: Catch vLLM crashes (like OOM errors) and surface them cleanlu
        raise HTTPException(status_code=500, detail=str(e))
