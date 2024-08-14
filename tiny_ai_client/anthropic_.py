from __future__ import annotations
import os
from typing import Any, Callable, Dict, List, Union, Generator, AsyncGenerator

from anthropic import Anthropic, AsyncAnthropic

from tiny_ai_client.models import LLMClientWrapper, Message, ToolCall
from tiny_ai_client.tools import function_to_json
from tiny_ai_client.vision import encode_pil_image


class AnthropicClientWrapper(LLMClientWrapper):
    def __init__(self, model_name: str, tools: List[Union[Callable, Dict]]):
        self.model_name = model_name
        self.client = Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )
        self.async_client = AsyncAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )
        self.tools = tools
        self.tools_json = [
            function_to_json(tool, "input_schema")["function"] for tool in tools
        ]

    def build_model_input(self, messages: List["Message"]) -> Any:
        input_messages = []
        system = None
        for message in messages:
            if message.role == "system":
                system = message.text
                continue
            elif message.tool_call:
                content_in = [
                    {
                        "type": "tool_use",
                        "name": message.tool_call.name,
                        "id": message.tool_call.id_,
                        "input": message.tool_call.parameters,
                    }
                ]
                content_out = [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call.id_,
                        "content": [
                            {"type": "text", "text": str(message.tool_call.result)}
                        ],
                    }
                ]
                input_messages.append(
                    {
                        "role": "assistant",
                        "content": content_in,
                    }
                )
                input_messages.append(
                    {
                        "role": "user",
                        "content": content_out,
                    }
                )
                input_messages.append(
                    {
                        "role": "assistant",
                        "content": "Acknowledged.",
                    }
                )
                continue
            else:
                content = []
                role = message.role
                if message.images:
                    for image in message.images:
                        content.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": encode_pil_image(image),
                                },
                            }
                        )
                if message.text is not None:
                    content.append(
                        {
                            "type": "text",
                            "text": message.text,
                        }
                    )
                model_input_message = {
                    "role": role,
                    "content": content,
                }
            input_messages.append(model_input_message)
        return input_messages, system

    def call_llm_provider(
        self,
        model_input: Any,
        temperature: int | None,
        max_new_tokens: int | None,
        timeout: int,
    ) -> str:
        kwargs = {}
        input_messages, system = model_input
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_new_tokens is not None:
            kwargs["max_tokens"] = max_new_tokens
        if system is not None:
            kwargs["system"] = system
        if self.tools_json:
            kwargs["tools"] = self.tools_json
        response = self.client.with_options(timeout=timeout).messages.create(
            model=self.model_name,
            messages=input_messages,
            **kwargs,
        )
        response_content = response.content[0]
        if response_content.type == "text":
            return Message(text=response_content.text, role="assistant")

        if response_content.type == "tool_use":
            print(response_content)
            return Message(
                role="assistant",
                text=None,
                tool_call=ToolCall(
                    id_=response_content.id,
                    parameters=response_content.input,
                    name=response_content.name,
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
        input_messages, system = self.build_model_input(chat)
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_new_tokens is not None:
            kwargs["max_tokens"] = max_new_tokens
        if system is not None:
            kwargs["system"] = system
        if self.tools_json:
            kwargs["tools"] = self.tools_json

        with self.client.with_options(timeout=timeout).messages.stream(
            model=self.model_name,
            messages=input_messages,
            **kwargs,
        ) as stream:
            for text in stream.text_stream:
                yield text

    async def async_call_llm_provider(
        self,
        model_input: Any,
        temperature: int | None,
        max_new_tokens: int | None,
        timeout: int,
    ) -> str:
        kwargs = {}
        input_messages, system = model_input
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_new_tokens is not None:
            kwargs["max_tokens"] = max_new_tokens
        if system is not None:
            kwargs["system"] = system
        if self.tools_json:
            kwargs["tools"] = self.tools_json
        response = await self.async_client.with_options(
            timeout=timeout
        ).messages.create(
            model=self.model_name,
            messages=input_messages,
            **kwargs,
        )
        response_content = response.content[0]
        if response_content.type == "text":
            return Message(text=response_content.text, role="assistant")

        if response_content.type == "tool_use":
            return Message(
                role="assistant",
                text=None,
                tool_call=ToolCall(
                    id_=response_content.id,
                    parameters=response_content.input,
                    name=response_content.name,
                ),
            )

    async def astream(
        self,
        max_new_tokens: int | None,
        temperature: int | None,
        timeout: int,
        chat: List["Message"],
    ) -> AsyncGenerator[str, None]:
        kwargs = {}
        input_messages, system = self.build_model_input(chat)
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_new_tokens is not None:
            kwargs["max_tokens"] = max_new_tokens
        if system is not None:
            kwargs["system"] = system
        if self.tools_json:
            kwargs["tools"] = self.tools_json

        async with self.async_client.with_options(timeout=timeout).messages.stream(
            model=self.model_name,
            messages=input_messages,
            **kwargs,
        ) as stream:
            async for text in stream.text_stream:
                yield text
