from __future__ import annotations

import os
from typing import Callable, Dict, List, Union

from openai import AsyncOpenAI, OpenAI

from tiny_ai_client.openai_ import OpenAIClientWrapper
from tiny_ai_client.tools import function_to_json


class GeminiClientWrapper(OpenAIClientWrapper):
    def __init__(self, model_name: str, tools: List[Union[Callable, Dict]]):
        assert (
            "googleapis.com" in os.environ["OPENAI_BASE_URL"]
        ), "OPENAI_BASE_URL must be set to inside googleapis.com"
        self.model_name = model_name
        self.client = OpenAI()
        self.async_client = AsyncOpenAI()
        self.tools = tools
        self.tools_json = [function_to_json(tool) for tool in tools]
