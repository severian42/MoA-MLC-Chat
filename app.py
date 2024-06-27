import gradio as gr
import os
from dotenv import load_dotenv
from utils import MLCEngineManager, generate_with_references, generate_mlc_stream, generate_mlc, inject_references_to_messages, get_model_engine
from functools import partial
from rich import print
from mlc_llm import MLCEngine
from huggingface_hub import snapshot_download
import json
import atexit

load_dotenv()

# Set environment variables for MLC-LLM
MLC_LLM_ENGINE_MODE = os.getenv("MLC_LLM_ENGINE_MODE")
MLC_LLM_MAX_BATCH_SIZE = os.getenv("MLC_LLM_MAX_BATCH_SIZE")
MLC_LLM_MAX_KV_CACHE_SIZE = os.getenv("MLC_LLM_MAX_KV_CACHE_SIZE")

ROUNDS = int(os.getenv("ROUNDS"))
MULTITURN = os.getenv("MULTITURN", "True") == "True"

MODEL_AGGREGATE = os.getenv("MODEL_AGGREGATE",)
MODEL_REFERENCE_1 = os.getenv("MODEL_REFERENCE_1")
MODEL_REFERENCE_2 = os.getenv("MODEL_REFERENCE_2")
MODEL_REFERENCE_3 = os.getenv("MODEL_REFERENCE_3")

default_reference_models = [MODEL_REFERENCE_1, MODEL_REFERENCE_2, MODEL_REFERENCE_3]

initialized_models = {}

engine_manager = MLCEngineManager()

def initialize_models():
    global MODEL_AGGREGATE, MODEL_REFERENCE_1, MODEL_REFERENCE_2, MODEL_REFERENCE_3
    models = [MODEL_AGGREGATE, MODEL_REFERENCE_1, MODEL_REFERENCE_2, MODEL_REFERENCE_3]
    
    for model in models:
        if model not in initialized_models:
            engine = get_model_engine(model)
            if engine is None:
                print(f"Failed to initialize model: {model}")
            else:
                print(f"Successfully initialized model: {model}")
                initialized_models[model] = engine

def moa_generate(messages, aggregate_model, reference_models, rounds, system_prompt):
    log_output = ""
    
    # Add system prompt to messages if provided
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages
    
    data = {
        "instruction": [messages] * len(reference_models),
        "references": [""] * len(reference_models),
        "model": reference_models,
    }

    for round in range(rounds):
        log_output += f"\n--- Round {round + 1} ---\n"
        
        results = [generate_with_references(model, data["instruction"][i], data["references"]) for i, model in enumerate(reference_models)]
        
        for i, result in enumerate(results):
            if result is None:
                log_output += f"\n{reference_models[i]}:\nError: Unable to generate response\n"
                data["references"][i] = f"Error with {reference_models[i]}"
            else:
                data["references"][i] = result
                log_output += f"\n{reference_models[i]}:\n{result}\n"
    
    aggregate_messages = inject_references_to_messages(messages, data["references"])

    try:
        output = ""
        for chunk in generate_mlc_stream(aggregate_model, aggregate_messages):
            output += chunk
        log_output += f"\n--- Aggregate Model ({aggregate_model}) ---\n{output}\n"
        return output, log_output
    except Exception as e:
        error_message = f"Error with aggregate model: {str(e)}"
        log_output += f"\n{error_message}\n"
        return error_message, log_output

def process_fn(item):
    try:
        model = item["model"]
        messages = item["instruction"]
        
        output = generate_mlc(model, messages)
        
        return {"output": output}
    except Exception as e:
        error_message = f"Error in process_fn for model {model}: {str(e)}"
        print(error_message)
        return {"output": error_message}

def process_bot_response(message, history, aggregate_model, reference_models, rounds, multi_turn, system_prompt):
    if not multi_turn:
        history = []
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
    
    messages.append({"role": "user", "content": message})
    
    output, log_output = moa_generate(messages, aggregate_model, reference_models, rounds, system_prompt)
    
    new_history = history + [[message, output]]
    return new_history, log_output


def create_gradio_interface():
    # Custom theme with earth tones
    theme = gr.themes.Base(
        primary_hue="green",
        secondary_hue="stone",
        neutral_hue="gray",
        font=("Helvetica", "sans-serif"),
    ).set(
        body_background_fill="linear-gradient(to right, #2c5e1a, #4a3728)",
        body_background_fill_dark="linear-gradient(to right, #1a3c0f, #2e2218)",
        button_primary_background_fill="#4a3728",
        button_primary_background_fill_hover="#5c4636",
        block_title_text_color="#e0d8b0",
        block_label_text_color="#c1b78f",
    )

    with gr.Blocks(theme=theme) as demo:
        gr.Markdown(
            """
            <div style="text-align: center;">
            
            # Mixture of Agents (MoA) Chat
            
            Welcome to the future of AI-powered conversations! This app combines multiple AI models
            to generate responses, merging their strengths for more accurate and diverse outputs.
            
            </div>
            """,
            elem_id="centered-markdown"
        )        
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Accordion("Model Configuration", open=True):
                    aggregate_model = gr.Dropdown(
                        choices=[MODEL_AGGREGATE, MODEL_REFERENCE_1, MODEL_REFERENCE_2, MODEL_REFERENCE_3],
                        value=MODEL_AGGREGATE,
                        label="Aggregate Model"
                    )
                    reference_models_box = gr.CheckboxGroup(
                        choices=[MODEL_REFERENCE_1, MODEL_REFERENCE_2, MODEL_REFERENCE_3],
                        value=default_reference_models,
                        label="Reference Models"
                    )
                
                with gr.Accordion("Generation Parameters", open=True):
                    rounds = gr.Slider(minimum=1, maximum=5, step=1, value=int(ROUNDS), label="Rounds")
                    multi_turn = gr.Checkbox(value=MULTITURN, label="Multi-turn Conversation")
                    system_prompt = gr.Textbox(
                        value="You are a helpful AI assistant.",
                        label="System Prompt",
                        lines=2
                    )
            
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Chat History", height=400)
                msg = gr.Textbox(label="Your Message", placeholder="Type your message here and press Enter...", lines=2)
                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear Chat")
        
        with gr.Accordion("Logs", open=False):
            logs = gr.Textbox(label="Processing Logs", lines=10)
        
        gr.Markdown(
            """
            ### How it works
            1. Your message is sent to multiple reference models.
            2. Each model generates a response.
            3. The aggregate model combines these responses to create a final output.
            4. The process repeats for the specified number of rounds.
            
            This approach allows for more nuanced and well-rounded responses!
            """
        )

        def send_message(message, history):
            return "", history + [[message, None]]

        def clear_chat():
            return None, None

        msg.submit(process_bot_response, [msg, chatbot, aggregate_model, reference_models_box, rounds, multi_turn, system_prompt], [chatbot, logs])
        send_btn.click(process_bot_response, [msg, chatbot, aggregate_model, reference_models_box, rounds, multi_turn, system_prompt], [chatbot, logs])

        clear_btn.click(clear_chat, outputs=[chatbot, logs])

    return demo

def cleanup():
    engine_manager.cleanup()

atexit.register(cleanup)

if __name__ == "__main__":
    initialize_models()
    demo = create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=4242)
