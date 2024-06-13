import asyncio

from tiny_ai_client import AI, AsyncAI


async def async_ai_main():
    print("### ASYNC AI ###")
    ai = AsyncAI(
        model_name="ollama:llama3",
        system="You are Spock, from Star Trek.",
        max_new_tokens=128,
        model_server_url="http://localhost:11434/api/chat",
    )
    response = await ai("What is the meaning of life?")
    print(f"{response=}")
    response = await ai("Did it work?")
    print(f"{response=}")
    print(f"{ai.chat=}")


def main():
    print("### SYNC AI ###")
    ai = AI(
        model_name="ollama:llama3",
        system="You are Spock, from Star Trek.",
        max_new_tokens=128,
        model_server_url="http://localhost:11434/api/chat",
        # tools=[get_current_weather],
    )
    response = ai("What is the meaning of life?")
    print(f"{response=}")
    response = ai("Did it work?")
    print(f"{response=}")
    print(f"{ai.chat=}")


if __name__ == "__main__":
    main()
    asyncio.run(async_ai_main())
