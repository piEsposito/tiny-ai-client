import os
from copy import deepcopy
from typing import Any, Callable, Dict, List, Union

import google.generativeai as genai

from tiny_ai_client.models import LLMClientWrapper, Message
from tiny_ai_client.tools import function_to_json


class GeminiClientWrapper(LLMClientWrapper):
    def __init__(self, model_name: str, tools: List[Union[Callable, Dict]]):
        self.model_name = model_name
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.tools = tools
        self.tools_json = [function_to_json(tool) for tool in tools]
        if len(self.tools_json) > 0:
            raise ValueError("Gemini does not support tools")
        # self.tools_obj = [
        #     genai.types.FunctionDeclaration(
        #         name=tool["function"]["name"],
        #         description=tool["function"]["description"],
        #         parameters=genai.types.content_types.strip_titles(
        #             tool["function"]["parameters"]
        #         ),
        #     )
        #     for tool in self.tools_json
        # ]

    def build_model_input(self, messages: List["Message"]) -> Any:
        history = []
        local_messages = deepcopy(messages)
        system = None
        message = None

        for message in local_messages:
            if message.role == "system":
                system = message.text
                continue
            else:
                if message.role not in ["user", "assistant"]:
                    raise ValueError(f"Invalid role for Gemini: {message.role}")
                role = "user" if message.role == "user" else "model"
                parts = []
                if message.text is not None:
                    parts.append(message.text)
                if message.images is not None:
                    parts.extend(message.images)
                history.append(
                    {
                        "role": role,
                        "parts": parts,
                    }
                )

        return (system, history)

    def call_llm_provider(
        self,
        model_input: Any,
        temperature: int | None,
        max_new_tokens: int | None,
        timeout: int,
    ) -> str:
        system, history = model_input

        generation_config_kwargs = {}
        if temperature is not None:
            generation_config_kwargs["temperature"] = temperature
        if max_new_tokens is not None:
            generation_config_kwargs["max_output_tokens"] = max_new_tokens

        generation_config = genai.GenerationConfig(**generation_config_kwargs)
        model = genai.GenerativeModel(
            self.model_name,
            system_instruction=system,
            generation_config=generation_config,
        )

        model.start_chat(history=history)
        response = model.generate_content(history)
        response = response.candidates[0].content.parts[0]
        if response.function_call.name != "":
            raise ValueError("Function calls are not supported in Gemini")
        elif response.text is not None:
            return Message(role="assistant", text=response.text)
        raise ValueError("Invalid response from Gemini")

    async def async_call_llm_provider(
        self,
        model_input: Any,
        temperature: int | None,
        max_new_tokens: int | None,
        timeout: int,
    ) -> str:
        system, history = model_input

        generation_config_kwargs = {}
        if temperature is not None:
            generation_config_kwargs["temperature"] = temperature
        if max_new_tokens is not None:
            generation_config_kwargs["max_output_tokens"] = max_new_tokens

        generation_config = genai.GenerationConfig(**generation_config_kwargs)
        model = genai.GenerativeModel(
            self.model_name,
            system_instruction=system,
            generation_config=generation_config,
        )

        model.start_chat(history=history)
        response = await model.generate_content_async(history)
        response = response.candidates[0].content.parts[0]
        if response.function_call.name != "":
            raise ValueError("Function calls are not supported in Gemini")
        elif response.text is not None:
            return Message(role="assistant", text=response.text)
        raise ValueError("Invalid response from Gemini")
