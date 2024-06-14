# Tiny AI Client

Inspired by [tinygrad](https://github.com/tinygrad/tinygrad) and [simpleaichat](https://github.com/minimaxir/simpleaichat/tree/main/simpleaichat), `tiny-ai-client` is the easiest way to use and switch LLMs with vision and tool usage support. It works because it is `tiny`, `simple` and most importantly `fun` to develop.

I want to change LLMs with ease, while knowing what is happening under the hood. Langchain is cool, but became bloated, complicated there is just too much chaos going on. I want to keep it simple, easy to understand and easy to use. If you want to use a LLM and have an API key, you should not need to read a 1000 lines of code and write `response.choices[0].message.content` to get the response.

Simple and tiny, that's the goal.

Features:

- OpenAI
- Anthropic
- Async
- Tool usage
- Structured output
- Vision
- PyPI package `tiny-ai-client`
- Gemini (vision, no tools)
- Ollama (text, no vision, no tools) (you can also pass a custom model_server_url to AI/AsyncAI)
  - To use it, `model_name='ollama:llama3'` or your model name.
- Groq (text, tools, no vision)
  - To use it `model_name='groq:llama-70b-8192'` or your model name as in Groq docs.

Roadmap:

- Gemini tools

## Simple

`tiny-ai-client` is simple and intuitive:

- Do you want set your model? Just pass the model name.
- Do you want to change your model? Just change the model name.
- Want to send a message? `msg: str = ai("hello")` and say goodbye to parsing a complex json.
- Do you want to use a tool? Just pass the tool as a function
  - Type hint it with a single argument that inherits from `pydantic.BaseModel` and just pass the callable. `AI` will call it and get its results to you if the model wants to.
- Want to use vision? Just pass a `PIL.Image.Image`.
- Video? Just pass a list of `PIL.Image.Image`.

## Tiny

- `tiny-ai-client` is very small, its core logic is < 250 lines of code (including comments and docstrings) and ideally won't pass 500. It is and always will be easy to understand, tweak and use.
  - The core logic is in `tiny_ai_client/models.py`
  - Vision utils are in `tiny_ai_client/vision.py`
  - Tool usage utils are in `tiny_ai_client/tools.py`
- The interfaces are implemented by subclassing `tiny_ai_client.models.LLMClientWrapper` binding it to a specific LLM provider. This logic might get bigger, but it is isolated in a single file and does not affect the core logic.

## Usage

```bash
pip install tiny-ai-client
```

To test, set the following environment variables:

- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- GROQ_API_KEY
- GOOGLE_API_KEY

Then

To run all examples:

```bash
./scripts/run-all-examples.sh
```

For OpenAI:

```python
from tiny_ai_client import AI, AsyncAI

ai = AI(
    model_name="gpt-4o", system="You are Spock, from Star Trek.", max_new_tokens=128
)
response = ai("What is the meaning of life?")

ai = AsyncAI(
    model_name="gpt-4o", system="You are Spock, from Star Trek.", max_new_tokens=128
)
response = await ai("What is the meaning of life?")
```

For Anthropic:

```python
from tiny_ai_client import AI, AsyncAI

ai = AI(
    model_name="claude-3-haiku-20240307", system="You are Spock, from Star Trek.", max_new_tokens=128
)
response = ai("What is the meaning of life?")

ai = AsyncAI(
    model_name="claude-3-haiku-20240307", system="You are Spock, from Star Trek.", max_new_tokens=128
)
response = await ai("What is the meaning of life?")
```

We also support tool usage for both. You can pass as many functions you want as type-hinted functions with a single argument that inherits from `pydantic.BaseModel`. `AI` will call the function and get its results to you.

```python
from pydantic import BaseModel, Field

from tiny_ai_client import AI, AsyncAI


class WeatherParams(BaseModel):
    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")
    unit: str = Field(
        "celsius", description="Temperature unit", enum=["celsius", "fahrenheit"]
    )


def get_current_weather(weather: WeatherParams):
    """
    Get the current weather in a given location
    """
    return f"Getting the current weather in {weather.location} in {weather.unit}."

ai = AI(
    model_name="gpt-4o",
    system="You are Spock, from Star Trek.",
    max_new_tokens=32,
    tools=[get_current_weather],
)
response = ai("What is the meaning of life?")
print(f"{response=}")
response = ai("Please get the current weather in celsius for San Francisco.")
print(f"{response=}")
response = ai("Did it work?")
print(f"{response=}")
```

And vision. Pass a list of `PIL.Image.Image` (or a single one) and we will handle the rest.

```python
from tiny_ai_client import AI, AsyncAI
from PIL import Image

ai = AI(
    model_name="gpt-4o",
    system="You are Spock, from Star Trek.",
    max_new_tokens=32,
)

response = ai(
    "Who is on the images?",
    images[
        Image.open("assets/kirk.jpg"),
        Image.open("assets/spock.jpg")
    ]
)
print(f"{response=}")

```
