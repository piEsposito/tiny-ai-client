import asyncio
import os

from PIL import Image as PIL_Image
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
    return {
        "abc": f"Getting the current weather in {weather.location} in {weather.unit}."
    }


def get_images():
    return [PIL_Image.open("assets/kirk.jpg"), PIL_Image.open("assets/spock.jpg")]


def main():
    print("### SYNC AI ###")
    ai = AI(
        model_name="gemini-1.5-flash",
        system="You are Spock, from Star Trek.",
        max_new_tokens=128,
        tools=[],
    )
    response = ai("How are you?")
    print(f"{response=}")
    response = ai("Who is on the images?", images=get_images())
    print(f"{response=}")
    # print(f"{ai.chat=}")


async def async_ai_main():
    print("### ASYNC AI ###")
    ai = AsyncAI(
        model_name="gemini-1.5-flash",
        system="You are Spock, from Star Trek.",
        max_new_tokens=128,
        tools=[],
    )
    response = await ai("How are you?")
    print(f"{response=}")
    response = await ai("Who is on the images?", images=get_images())
    print(f"{response=}")
    # print(f"{ai.chat=}")


if __name__ == "__main__":
    os.environ["GOOGLE_API_KEY"] = ""
    main()
    asyncio.run(async_ai_main())
