from __future__ import annotations
from typing import Any, Callable, Dict, List, Union

import aiohttp
import requests

from tiny_ai_client.models import LLMClientWrapper, Message


class OllamaClientWrapper(LLMClientWrapper):
    def __init__(
        self,
        model_name: str,
        tools: List[Union[Callable, Dict]],
        url: str = "http://localhost:11434/api/chat",
    ):
        self.model_name = model_name.split("ollama:")[1]
        self.url = url
        if tools:
            raise ValueError("Ollama does not support tools")

    def build_model_input(self, messages: List["Message"]) -> Any:
        input_messages = []
        for message in messages:
            if message.tool_call:
                raise ValueError("Ollama does not support tool calls")
            else:
                if message.text is not None:
                    content = message.text
                if message.images:
                    raise ValueError("Ollama does not support images")
                model_input_message = {
                    "role": message.role,
                    "content": content,
                }
                input_messages.append(model_input_message)
        model_input = {
            "messages": input_messages,
            "model": self.model_name,
            "stream": False,
        }
        return model_input

    def call_llm_provider(
        self,
        model_input: Any,
        temperature: int | None,
        max_new_tokens: int | None,
        timeout: int,
    ) -> str:
        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_new_tokens is not None:
            kwargs["num_ctx"] = max_new_tokens
        model_input["options"] = kwargs
        response = requests.post(self.url, json=model_input, timeout=timeout)
        response = response.json()
        chat_response = response["message"]["content"]
        if chat_response is not None:
            return Message(text=chat_response, role="assistant")

    async def async_call_llm_provider(
        self,
        model_input: Any,
        temperature: int | None,
        max_new_tokens: int | None,
        timeout: int,
    ) -> str:
        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_new_tokens is not None:
            kwargs["num_ctx"] = max_new_tokens
        model_input["options"] = kwargs
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.url, json=model_input, timeout=timeout
            ) as response:
                response_data = await response.json()
                chat_response = response_data["message"]["content"]
                if chat_response is not None:
                    return Message(text=chat_response.strip(), role="assistant")
