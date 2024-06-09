from typing import Any, Callable, Dict, List, Union

from PIL import Image as PIL_Image
from pydantic import BaseModel

from tiny_ai_client.tools import json_to_function_input

"""
Using the factory design pattern approach
provides a way to encapsulate the instantiate-and-return 
The logic is code easy to code and test, much cleaner
Following the best object-oriented design practices (DRI, SRP, OCP, Factory).
"""


class ClientWrapperFactory:
    _client_wrapper_mapping = {
        "gpt": "tiny_ai_client.openai_.OpenAIClientWrapper",
        "claude": "tiny_ai_client.anthropic_.AnthropicClientWrapper",
        "gemini": "tiny_ai_client.gemini_.GeminiClientWrapper"
    }

    @staticmethod
    def create(model_name: str, tools: List[Union[Callable, Dict]]) -> "LLMClientWrapper":
        for key, wrapper_path in ClientWrapperFactory._client_wrapper_mapping.items():
            if key in model_name:
                module_path, wrapper_class = wrapper_path.rsplit('.', 1)
                module = __import__(module_path, fromlist=[wrapper_class])
                return getattr(module, wrapper_class)(model_name, tools)
        raise NotImplementedError(f"{model_name=} not supported")


class AI:
    def __init__(
            self,
            model_name: str,
            system: str | None = None,
            temperature: int = 1,
            max_new_tokens: int | None = None,
            timeout: int = 30,
            tools: List[Union[Callable, Dict]] | None = None,
            chat: List["Message"] | None = None,
    ):
        # llm sampling parameters
        self.temperature: int = temperature
        self.max_new_tokens: int | None = max_new_tokens
        self.timeout: int = timeout
        self.chat: List[Message] = chat or []
        if system:
            self.chat.append(Message(text=system, role="system"))

        tools = tools or []
        self.tools = tools
        self.tools_dict = {tool.__name__: tool for tool in tools}

        self.model_name: str = model_name
        self.system: str = system
        self.client_wrapper: LLMClientWrapper = self.get_llm_client_wrapper(
            model_name=model_name, tools=self.tools
        )

    def __call__(
            self,
            message: str | None = None,
            max_new_tokens: int | None = None,
            temperature: int | None = 1,
            timeout: int | None = None,
            images: List[PIL_Image.Image] | PIL_Image.Image | None = None,
    ) -> str:
        max_new_tokens = max_new_tokens or self.max_new_tokens
        temperature = temperature or self.temperature
        timeout = timeout or self.timeout
        message = message or ""
        if isinstance(images, PIL_Image.Image):
            images = [images]
        self.chat.append(Message(text=message, role="user", images=images))
        response_msg: "Message" = self.client_wrapper(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            chat=self.chat,
            timeout=timeout,
        )
        self.chat.append(response_msg)
        if response_msg.tool_call:
            func = self.tools_dict[response_msg.tool_call.name]
            tool_input = json_to_function_input(func, response_msg.tool_call.parameters)
            tool_result = func(tool_input)
            response_msg.tool_call.result = tool_result
        return response_msg.text or (
            response_msg.tool_call.result if response_msg.tool_call else ""
        )

    # noinspection PyMethodMayBeStatic
    def get_llm_client_wrapper(
            self, model_name: str, tools: List[Union[Callable, Dict]]
    ) -> "LLMClientWrapper":
        return ClientWrapperFactory.create(model_name, tools)

    @property
    def model_name(self) -> str:
        return self.model_name

    @model_name.setter
    def model_name(self, value: str) -> None:
        self.model_name = value
        self.client_wrapper = self.get_llm_client_wrapper(value, self.tools)


class AsyncAI(AI):
    async def __call__(
            self,
            message: str | None = None,
            max_new_tokens: int | None = None,
            temperature: int | None = 1,
            timeout: int | None = None,
            images: List[PIL_Image.Image] | PIL_Image.Image | None = None,
    ) -> str:
        max_new_tokens = max_new_tokens or self.max_new_tokens
        temperature = temperature or self.temperature
        timeout = timeout or self.timeout
        message = message or ""
        if isinstance(images, PIL_Image.Image):
            images = [images]
        self.chat.append(Message(text=message, role="user", images=images))
        response_msg: "Message" = await self.client_wrapper.acall(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            chat=self.chat,
            timeout=timeout,
        )
        self.chat.append(response_msg)
        if response_msg.tool_call:
            func = self.tools_dict[response_msg.tool_call.name]
            tool_input = json_to_function_input(func, response_msg.tool_call.parameters)
            tool_result = func(tool_input)
            response_msg.tool_call.result = tool_result
        return response_msg.text or (
            response_msg.tool_call.result if response_msg.tool_call else ""
        )


class ToolCall(BaseModel):
    id_: str
    parameters: Dict[str, Any]
    result: str | None = None
    name: str

    def format(self) -> str:
        return f"Called {self.name} with ({self.parameters}) and got {self.result}"


class Message(BaseModel):
    text: str | None = None
    role: str
    tool_call: ToolCall | None = None
    images: List[PIL_Image.Image] | None = None

    class Config:
        arbitrary_types_allowed = True


class LLMClientWrapper:
    def build_model_input(self, messages: List[Message]) -> Any:
        raise NotImplementedError

    def call_llm_provider(
            self,
            model_input: Any,
            temperature: int | None,
            max_new_tokens: int | None,
            timeout: int,
    ) -> Message:
        raise NotImplementedError

    def __call__(
            self,
            max_new_tokens: int | None,
            temperature: int | None,
            timeout: int,
            chat: List[Message],
    ) -> Message:
        model_input = self.build_model_input(chat)
        return self.call_llm_provider(
            model_input,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            timeout=timeout,
        )

    async def async_call_llm_provider(
            self,
            model_input: Any,
            temperature: int | None,
            max_new_tokens: int | None,
            timeout: int,
    ) -> Message:
        raise NotImplementedError

    async def acall(
            self,
            max_new_tokens: int | None,
            temperature: int | None,
            timeout: int,
            chat: List[Message],
    ) -> Message:
        model_input = self.build_model_input(chat)
        return await self.async_call_llm_provider(
            model_input,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            timeout=timeout,
        )
