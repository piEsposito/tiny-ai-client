from __future__ import annotations
import json
from typing import Any, AsyncGenerator, Callable, Dict, Generator, List, Union

from groq import AsyncGroq, Groq

from tiny_ai_client.models import LLMClientWrapper, Message, ToolCall
from tiny_ai_client.tools import function_to_json


class GroqClientWrapper(LLMClientWrapper):
    def __init__(self, model_name: str, tools: List[Union[Callable, Dict]]):
        self.model_name = model_name.split("groq:")[1]
        self.client = Groq()
        self.async_client = AsyncGroq()
        self.tools = tools
        self.tools_json = [function_to_json(tool) for tool in tools]

    def build_model_input(self, messages: List["Message"]) -> Any:
        input_messages = []
        for message in messages:
            if message.tool_call is not None:
                tool_call_in = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": message.tool_call.id_,
                            "function": {
                                "name": message.tool_call.name,
                                "arguments": json.dumps(message.tool_call.parameters),
                            },
                            "type": "function",
                        }
                    ],
                }
                tool_call_response = {
                    "tool_call_id": message.tool_call.id_,
                    "role": "tool",
                    "name": message.tool_call.name,
                    "content": str(message.tool_call.result),
                }
                input_messages.extend([tool_call_in, tool_call_response])
                continue
            else:
                content = None
                assert message.text is not None, "Message text cannot be None for Groq"
                content = message.text
                if message.images:
                    raise ValueError("Groq does not support images yet.")
                model_input_message = {
                    "role": message.role,
                    "content": content,
                }
                input_messages.append(model_input_message)
        return input_messages

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
            kwargs["max_tokens"] = max_new_tokens
        if self.tools_json:
            kwargs["tools"] = self.tools_json
        response = self.client.with_options(timeout=timeout).chat.completions.create(
            model=self.model_name,
            messages=model_input,
            **kwargs,
        )
        chat_response = response.choices[0].message
        if chat_response.content is not None:
            return Message(text=chat_response.content.strip(), role="assistant")
        if chat_response.tool_calls:
            return Message(
                role="assistant",
                text=None,
                tool_call=ToolCall(
                    id_=chat_response.tool_calls[0].id,
                    parameters=json.loads(
                        chat_response.tool_calls[0].function.arguments
                    ),
                    name=chat_response.tool_calls[0].function.name,
                ),
            )

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
            kwargs["max_tokens"] = max_new_tokens
        if self.tools_json:
            kwargs["tools"] = self.tools_json
        response = await self.async_client.with_options(
            timeout=timeout
        ).chat.completions.create(
            model=self.model_name,
            messages=model_input,
            **kwargs,
        )
        chat_response = response.choices[0].message
        if chat_response.content is not None:
            return Message(text=chat_response.content.strip(), role="assistant")
        if chat_response.tool_calls:
            return Message(
                role="assistant",
                text=None,
                tool_call=ToolCall(
                    id_=chat_response.tool_calls[0].id,
                    parameters=json.loads(
                        chat_response.tool_calls[0].function.arguments
                    ),
                    name=chat_response.tool_calls[0].function.name,
                ),
            )

    def stream(
        self,
        max_new_tokens: int | None,
        temperature: int | None,
        timeout: int,
        chat: List["Message"],
    ) -> Generator[str, None, None]:
        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_new_tokens is not None:
            kwargs["max_tokens"] = max_new_tokens

        model_input = self.build_model_input(chat)

        stream = self.client.with_options(timeout=timeout).chat.completions.create(
            model=self.model_name,
            messages=model_input,
            stream=True,
            **kwargs,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    async def astream(
        self,
        max_new_tokens: int | None,
        temperature: int | None,
        timeout: int,
        chat: List["Message"],
    ) -> AsyncGenerator[str, None]:
        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_new_tokens is not None:
            kwargs["max_tokens"] = max_new_tokens

        model_input = self.build_model_input(chat)

        stream = await self.async_client.with_options(
            timeout=timeout
        ).chat.completions.create(
            model=self.model_name,
            messages=model_input,
            stream=True,
            **kwargs,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
