import asyncio
import json
import uuid

from typing import AsyncGenerator
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from transformers import AutoTokenizer

from .models import ChatCompletionRequest

class ForgeEngine:
    """
    Wrapper around vLLM's AsyncLLMEngine.
    
    DESIGN DECISION: Why wrap vLLM instead of using it directly?
    1. Abstraction: It decouples our FastAPI routing layer from vLLM's specific API. 
       If we want to test a custom continuous batching scheduler later (Phase 2), 
       we only have to change this file, not our web routes.
    2. Resource Control: We explicitly lock down VRAM allocation here.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        # Configure vLLM engine arguments
        engine_args = AsyncEngineArgs(
            
            model=self.model_name,
            # TENSOR PARALLELISM: Set to 1 because you are running a single RTX 5080.
            # Implication: The model weights must fit entirely inside 16GB of VRAM.
            tensor_parallel_size=1, # TO Change if we are using multiple GPUs
            
            
            # MEMORY ALLOCATION: We allocate 80% of your 16GB VRAM to vLLM.
            # Implication: vLLM will reserve ~14.4GB immediately upon startup. 
            # The remaining 10% is left as a safety buffer for PyTorch context 
            # overhead and the OS, preventing sudden CUDA Out-Of-Memory (OOM) crashes.
            gpu_memory_utilization=0.80, # Reserve 80% of the RTX 5080's VRAM

            max_model_len=4096, # Cap context to 4K tokens to save KV-Cache VRAM
            
            # EAGER MODE: False means we allow vLLM to use CUDA Graph capture where possible.
            # Implication: Higher throughput and lower CPU overhead during generation.
            enforce_eager=False 
        )

        # Initialize the asynchronous engine. 
        # By using AsyncLLMEngine, the event loop isn't blocked while the GPU computes.
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        # Load the tokenizer explicitly for prompt formatting
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    async def stream_chat(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        messages_dict = [{"role": m.role, "content": m.content} for m in request.messages]
        
        """
        If we just sent the model a raw JSON array like [{"role": "user", "content": "Hello"}], the model would literally try to predict the next JSON bracket.

        To make a model act like a chatbot, the creators (Alibaba for Qwen) fine-tuned the model to respond to a highly specific script format. Qwen uses a format called "ChatML". apply_chat_template takes nice list of JSON messages and automatically injects the exact hidden tokens the model expects.

        It turns this:
        [{"role": "user", "content": "Hello"}]

        Into this string:
        <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n
        """
        prompt = self.tokenizer.apply_chat_template(
            messages_dict,
            tokenize = False, # give me back a readable string, not raw integer ids
            add_generation_prompt=True # It appends <|im_start|>assistant\n to the very end of the string. Without this the AI might hallucinate and just start generating more user messages!
        )
        
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        generator = self.engine.generate(prompt, sampling_params, request_id)
        
        # 2. Track text to calculate deltas
        previous_text = ""

        async for request_output in generator:
            # Extract only the newly generated characters
            current_text = request_output.outputs[0].text
            new_text = current_text[len(previous_text):]
            previous_text = current_text
            
            # 3. Pack into the OpenAI streaming chunk schema
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "model": request.model or "forge-model",
                "choices": [{
                    "index": 0,
                    "delta": {"content": new_text}
                }]
            }
            
            # SSE requirement: "data: {json_string}\n\n"
            yield f"data: {json.dumps(chunk)}\n\n"
            
        # 4. Signal that the stream is complete
        yield "data: [DONE]\n\n"




    async def generate(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        # 1. Convert Pydantic objects into standard dictionaries
        messages_dict = [{"role": m.role, "content": m.content} for m in request.messages]

        # 2. Apply the correct Chat Template (Replaces the naive string join)
        prompt = self.tokenizer.apply_chat_template(
            messages_dict,
            tokenize=False, 
            add_generation_prompt=True
        )

        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        ) 

        # 3. Generate Request ID
        request_id = random_uuid()

        results_generator = self.engine.generate(prompt, sampling_params, request_id)

        async for request_output in results_generator:
            yield request_output.outputs[0].text

    async def generate_DEPRECATED(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        # Convert Open AI format messages into a single prompt string
        """
        Takes a validated Pydantic request, formats the prompt, and yields tokens.
        """
        # DESIGN DECISION: Prompt Formatting
        # Currently doing a naive string join. 
        # Implication: This works for basic testing, but in production (Week 3/4), 
        # we MUST replace this with HuggingFace's `apply_chat_template` so we don't 
        # break the specific instruction formatting expected by models like Llama-3.
        prompt = "\n".join(f"{msg.role}: {msg.content}" for msg in request.messages)
        prompt += "\assistant: "

        # Map our OpenAI-compatible request parameters to vLLM's internal format
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        ) 

        request_id = random_uuid

        # The engine returns an asynchronous generator. 
        # This is the foundation of continuous batching: while this specific request 
        # is awaiting its next token, the engine is free to process other requests.
        results_generator = self.engine.generate(prompt, sampling_params, request_id)

        async for request_output in results_generator:
            # Yielding the raw text output. 
            # Note: vLLM returns the ENTIRE generated string so far on each yield.
            # When we build the SSE streaming router, we will need to calculate the 
            # delta (the newest token) to match the OpenAI streaming standard.
            yield request_output.outputs[0].text