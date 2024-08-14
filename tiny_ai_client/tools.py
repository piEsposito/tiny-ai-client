from __future__ import annotations
from typing import get_type_hints

from pydantic import BaseModel


def function_to_json(func, parameters_key="parameters") -> dict:
    func_name = func.__name__
    func_desc = func.__doc__.strip().split("\n")[0] if func.__doc__ else ""

    hints = get_type_hints(func)

    # Extract the BaseModel from the function's parameters
    param_name, param_type = next(iter(hints.items()))
    if not issubclass(param_type, BaseModel):
        raise ValueError("The function's parameter must be a Pydantic BaseModel")

    model_schema = param_type.schema()

    func_json = {
        "type": "function",
        "function": {
            "name": func_name,
            "description": func_desc,
            parameters_key: {
                "type": "object",
                "properties": model_schema["properties"],
                "required": model_schema.get("required", True),
            },
        },
    }
    return func_json


def json_to_function_input(func, json_input: dict):
    hints = get_type_hints(func)
    param_name, param_type = next(iter(hints.items()))
    return param_type(**json_input)
