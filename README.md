# Mixture-of-Agents (MoA) Chat Application

This is an advanced implementation of the Mixture-of-Agents (MoA) concept, adapted from the original work by Together Computer. Our version is tailored for local model usage and features a user-friendly Gradio interface.

## What is MoA?

Mixture of Agents (MoA) is a cutting-edge approach that leverages multiple Large Language Models (LLMs) to enhance AI performance. By utilizing a layered architecture where each layer consists of several LLM agents, MoA achieves state-of-the-art results using open-source models.

## Gradio Interface

![Gradio UI Screenshot](assets/gradioui.png)

## Key Features

- **Multi-Model Integration**: Combines responses from multiple AI models for more comprehensive and nuanced outputs.
- **Customizable Model Selection**: Users can choose and configure both reference and aggregate models.
- **Adjustable Parameters**: Fine-tune generation with customizable temperature, max tokens, and processing rounds.
- **Real-Time Streaming**: Experience fluid, stream-based response generation.
- **Intuitive Gradio Interface**: User-friendly UI with an earth-toned theme for a pleasant interaction experience.
- **Flexible Conversation Modes**: Support for both single-turn and multi-turn conversations.

## How It Works

1. User input is processed by multiple reference models simultaneously.
2. Each reference model generates its unique response.
3. An aggregate model combines and refines these responses into a final output.
4. This process can be repeated for multiple rounds, enhancing the quality of the final response.

## Setup and Configuration

1. Clone the repository and navigate to the project directory.

2. Set up your environment:

   ```shell
   cp .env.example .env
   vi .env
   ```

3. Install requirements:

   ```shell
   conda create -n moa python=3.10
   conda activate moa
   pip install -r requirements.txt
   ```

## Configuration

Edit the `.env` file to configure the following parameters:

```bash
MLC_LLM_ENGINE_MODE = server
MLC_LLM_MAX_BATCH_SIZE = 80
MLC_LLM_MAX_KV_CACHE_SIZE = 32768


ROUNDS=1

MODEL_AGGREGATE=HF://mlc-ai/Hermes-2-Theta-Llama-3-8B-q4f16_1-MLC
MODEL_REFERENCE_1=HF://mlc-ai/Qwen2-0.5B-Instruct-q4f16_1-MLC
MODEL_REFERENCE_2=HF://mlc-ai/Phi-3-mini-128k-instruct-q4f16_1-MLC
MODEL_REFERENCE_3=HF://mlc-ai/Qwen2-1.5B-Instruct-q4f32_1-MLC

MULTITURN=True
```

## Running the Application

1. Launch the Gradio interface:

   ```shell
   conda activate moa
   gradio app.py
   ```

3. Open your web browser and navigate to the URL http://localhost:4242.

## Advanced Usage

- **Model Customization**: Easily switch between different reference models to suit your needs.
- **Parameter Tuning**: Adjust rounds to control the output's creativity and length.
- **Multi-Turn Conversations**: Enable or disable context retention for more dynamic interactions.

## Performance Insights

While specific benchmarks are not provided, the MoA approach has shown significant improvements over single-model systems, potentially outperforming some commercial AI solutions in certain tasks.

## Contributing

We welcome contributions to enhance the MoA Chat Application. Feel free to submit pull requests or open issues for discussions on potential improvements.

## License

This project is licensed under the terms specified in the original MoA repository. Please refer to the original source for detailed licensing information.

---

<div align="center">
  <img src="assets/moa.jpg" alt="MoA Concept Visualization" style="width: 100%; max-width: 600px;" />
</div>
