import os
import json
import time
from mlc_llm import MLCEngine
from loguru import logger
from dotenv import load_dotenv
from contextlib import contextmanager
from huggingface_hub import snapshot_download

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE")

MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096")) 
TEMPERATURE = float(os.getenv("TEMPERATURE", "7"))

DEBUG = int(os.environ.get("DEBUG", "0"))

class MLCEngineManager:
    def __init__(self):
        self.engines = {}

    @contextmanager
    def get_engine(self, model_name):
        if model_name not in self.engines:
            self.engines[model_name] = MLCEngine(
                model=model_name,
                mode="server",
            )
        engine = self.engines[model_name]
        try:
            yield engine
        finally:
            # Perform any necessary cleanup here
            pass

    def cleanup(self):
        for engine in self.engines.values():
            engine.unload()  # Assuming there's an unload method, adjust if necessary
        self.engines.clear()

engine_manager = MLCEngineManager()

def generate_mlc(model_name, messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS):
    with engine_manager.get_engine(model_name) as model:
        formatted_messages = []
        system_prompt = ""
        
        for msg in messages:
            if msg['role'] == 'system':
                system_prompt = msg['content']
            elif msg['role'] == 'user':
                if system_prompt:
                    formatted_messages.append({"role": "user", "content": f"{system_prompt}\n\nUser: {msg['content']}"})
                    system_prompt = ""
                else:
                    formatted_messages.append({"role": "user", "content": msg['content']})
            elif msg['role'] == 'assistant':
                formatted_messages.append({"role": "assistant", "content": msg['content']})

        request = {
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        response = model.chat.completions.create(**request)
        return response.choices[0].message.content.strip()

def generate_mlc_stream(model_name, messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS):
    with engine_manager.get_engine(model_name) as model:
        formatted_messages = []
        system_prompt = ""
        
        for msg in messages:
            if msg['role'] == 'system':
                system_prompt = msg['content']
            elif msg['role'] == 'user':
                if system_prompt:
                    formatted_messages.append({"role": "user", "content": f"{system_prompt}\n\nUser: {msg['content']}"})
                    system_prompt = ""
                else:
                    formatted_messages.append({"role": "user", "content": msg['content']})
            elif msg['role'] == 'assistant':
                formatted_messages.append({"role": "assistant", "content": msg['content']})
        request = {
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        for response in model.chat.completions.create(**request):
            if response.choices[0].delta.content is not None:
                yield response.choices[0].delta.content

def generate_with_references(model_name, messages, references=[], temperature=TEMPERATURE):
    if len(references) > 0:
        messages = inject_references_to_messages(messages, references)

    return generate_mlc(model_name, messages, temperature=temperature)

def inject_references_to_messages(messages, references):
    messages = messages.copy()
    system = f"""You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""

    for i, reference in enumerate(references):
        system += f"\n{i+1}. {reference}"

    messages = [{"role": "system", "content": system}] + messages

    return messages

def is_model_downloaded(model_id, models_dir):
    model_dir = os.path.join(models_dir, model_id.replace("/", "--"))
    config_file = os.path.join(model_dir, "mlc-chat-config.json")
    return os.path.exists(config_file)

def get_model_engine(model):
    try:
        if model.startswith("HF://"):
            model_id = model[5:]  
            models_dir = os.path.join(os.getcwd(), "mlc_models")
            os.makedirs(models_dir, exist_ok=True)
            
            if not is_model_downloaded(model_id, models_dir):
                print(f"Downloading model: {model_id}")
                local_dir = snapshot_download(
                    repo_id=model_id,
                    local_dir=os.path.join(models_dir, model_id.replace("/", "--")),
                    local_dir_use_symlinks=False,
                    allow_patterns=["*.json", "*.bin"],
                    ignore_patterns=["*.md", "*.txt"],
                )
            else:
                print(f"Model already downloaded: {model_id}")
                local_dir = os.path.join(models_dir, model_id.replace("/", "--"))
            
            model = local_dir

        engine = MLCEngine(
            model=model,
            device="auto",
            mode="server"
        )
        return engine
    except Exception as e:
        print(f"Error initializing model engine for {model}: {str(e)}")
        return None