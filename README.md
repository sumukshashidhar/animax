# animax

Your personal Animax Foundry

## Overview

`animax` provides the `Anima` class, a simple and extensible interface for interacting with LLMs (Large Language Models) via the [LiteLLM](https://github.com/BerriAI/litellm) backend. It is designed for conversational and single-turn completions, with robust error handling and easy customization.

## Installation

Install from PyPI (coming soon):

```bash
uv pip install animax
```

Or, for development:

```bash
git clone https://github.com/sumukshashidhar/animax.git
cd animax
uv pip install -e .
```

## Quickstart

### 1. Basic Usage

```python
from animax import Anima

# You must provide your API key for the backend (e.g., OpenAI, Azure, etc.)
anima = Anima(api_key="sk-...")  # Replace with your actual API key

response = anima.ask("What is the capital of France?")
print(response)  # Output: Paris
```

### 2. Conversational Chat

The `chat` method maintains conversation history:

```python
from animax import Anima

anima = Anima(api_key="sk-...")

print(anima.chat("Who won the FIFA World Cup in 2018?"))
print(anima.chat("And who was the top scorer?"))  # Context is preserved
```

### 3. Customization

You can customize the model, system prompt, and other parameters:

```python
anima = Anima(
    api_key="sk-...",
    model_name="gpt-3.5-turbo",
    system_prompt="You are a helpful assistant."
)

print(anima.ask("Summarize the plot of Inception."))
```

### 4. Backend Validation

Check if your API key and backend are working:

```python
if anima.validate_backend():
    print("Backend is valid!")
else:
    print("Backend validation failed.")
```

## Parameters

- `name`: Optional name for your anima instance.
- `backend`: Backend to use (default: `"litellm"`).
- `model_endpoint_style`: Endpoint style, e.g., `"openai"` (default).
- `model_name`: Model to use (default: `"gpt-4o"`).
- `api_key`: **Required**. Your LLM provider API key.
- `api_base`: Optional custom API base URL.
- `system_prompt`: Optional system prompt for persona/context.
- `previous_messages`: Optional initial conversation history.

## API Reference

### `Anima.ask(prompt: str, max_tokens: int = 4096, temperature: float = 1.0) -> str`

Ask a single-turn question.

### `Anima.chat(prompt: str, max_tokens: int = 4096, temperature: float = 1.0) -> str`

Chat with context/history.

### `Anima.validate_backend() -> bool`

Validate backend connectivity and credentials.

## Environment Variables

The package loads environment variables via `.env` (using `python-dotenv`), but you **must** pass your API key explicitly to the `Anima` constructor.

## License

See [LICENSE](LICENSE).

## Links

- [Homepage](https://github.com/sumukshashidhar/animax)
- [PyPI (coming soon)](https://pypi.org/project/animax/)

---

*For more advanced usage, see the source code and docstrings.*
