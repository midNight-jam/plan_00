import uuid
import time
import os

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from vllm import SamplingParams

from .models import (
    ChatCompletionRequest, 
    ChatCompletionResponse, 
    ChatCompletionResponseChoice,
    ChatMessage,
    HealthResponse
)
from .database import init_db, AsyncSessionLocal, RequestLog
# from fastapi import Request, HTTPException
from .engine import ForgeEngine
from .config import settings


# 1. Define the Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP LOGIC ---
    print("Starting up Forge API...")
    print("Initializing PostgreSQL database...")
    await init_db()
    print("Database initialized successfully.")
    
    # Yield control back to FastAPI to run the actual web server
    yield
    
    # --- SHUTDOWN LOGIC ---
    # Anything below the 'yield' runs when you hit Ctrl+C or kill the Docker container
    print("Shutting down Forge API...")
    # (In the future, we could put graceful vLLM shutdown logic here!)



app = FastAPI(title="zzForge Inference Server", lifespan=lifespan)

# INITIALIZATION :  instantiates the engine globally so it loads the model weights 
# into VRAM once upon server startup, rather than on every request.
# MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
# MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct") # this failed to loadin gpu :(, thus going for AWQ
# MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct-AWQ")
MODEL_NAME = settings["model"]["name"]
engine = ForgeEngine(model_name=MODEL_NAME)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest, background_tasks: BackgroundTasks): # Using Pydantic model
    start_time = time.time()
    try:

        # Route to the streaming generator if requested
        if request.stream:
            background_tasks.add_task(log_request_to_db, request.model or MODEL_NAME, str(request.messages), start_time)
            return StreamingResponse(
                engine.stream_chat(request),
                media_type="text/event-stream"
            )
            
        # Otherwise, fall back to your existing non-streaming logic

        #. Send the validated Pydantic model to the vLLM engine
        results_generator = engine.generate(request)
        
        # 6. Await the final result (for non-streaming)
        generated_text = ""
        final_output = None
        async for text in results_generator:
            generated_text = text # engine.py yields the text string directly
            
        # Log the non-streaming requests too!
        background_tasks.add_task(log_request_to_db, request.model or MODEL_NAME, str(request.messages), start_time)
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


@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        # In modern versions of vLLM, the AsyncLLMEngine has a check_health() method.
        # It will raise an exception if the background GPU worker thread crashed (e.g., CUDA OOM).
        if hasattr(engine.engine, "check_health"):
            engine.engine.check_health()
            
        return HealthResponse(
            status="healthy",
            model=MODEL_NAME
        )
    except Exception as e:
        # If the GPU worker died, we throw a 503 Service Unavailable
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail=f"Engine unhealthy: {str(e)}")


async def log_request_to_db(model_name: str, prompt: str, start_time: float):
    end_time = time.time()
    generation_time_ms = (end_time - start_time) * 1000
    
    async with AsyncSessionLocal() as session:
        log_entry = RequestLog(
            model_name=model_name,
            prompt_length=len(prompt),
            generation_time_ms=generation_time_ms
        )
        session.add(log_entry)
        await session.commit()