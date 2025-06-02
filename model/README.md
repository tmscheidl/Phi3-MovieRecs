# Models

This directory documents the model architecture and configuration used in the Movie Recommendation System.

## Model Overview

- **Model Name:** `microsoft/Phi-3.5-mini-instruct`
- **Source:** [Hugging Face Model Hub](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)
- **Model Type:** Instruction-tuned, decoder-only language model
- **Primary Use:** The model is employed to generate personalized movie recommendations based on the user's recent interaction history and a generated prompt in natural language.

The model is loaded dynamically at runtime using the Hugging Face `transformers` library. Model weights are not stored locally in this repository to reduce storage requirements and maintain versioning flexibility.

## Purpose of This Directory

The `models/` directory serves the following purposes:

- Provide documentation on the model used within the system
- Store configurations and metadata related to the pre-trained model
- Maintain reusable prompt templates or tokenizer-related settings
- Serve as a location for saving fine-tuned checkpoints (if applicable in the future)
- Track experimental results or logs related to model evaluation and ablation studies (if included)

## Model Loading Example

The following snippet illustrates how the model is initialized and loaded within the project codebase:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "microsoft/Phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

