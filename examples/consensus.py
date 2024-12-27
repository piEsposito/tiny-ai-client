import asyncio
from typing import List

from pydantic import BaseModel

from tiny_ai_client import AsyncAI


class Discusser(BaseModel):
    model: AsyncAI
    description: str | None

    model_config = {"arbitrary_types_allowed": True}


class ConsensusAI:
    def __init__(
        self,
        judge: AsyncAI,
        discussers: List[Discusser],
    ):
        self.judge = judge
        self.discussers = discussers

    async def get_consensus(self, prompt: str, **kwargs) -> str:
        # Gather responses from all discussers
        responses = []
        responses = await asyncio.gather(
            *[discusser.model(prompt, **kwargs) for discusser in self.discussers]
        )
        responses = [
            f"Expert {i+1}: {response}" for i, response in enumerate(responses)
        ]

        # Format responses for the judge
        discussion = "\n\n".join(responses)
        judge_prompt = f"""Here are expert responses to: "{prompt}"

{discussion}

Based on these responses, what is the consensus?"""

        return await self.judge(judge_prompt)


def main():
    # Initialize discussers with different models
    discussers = [
        Discusser(
            model=AsyncAI(
                model_name="gemini-2.0-flash-thinking-exp",
                system="You are a thoughtful expert.",
            ),
            description="Gemini Expert",
        ),
        Discusser(
            model=AsyncAI(
                model_name="claude-3-5-sonnet-20241022",
                system="You are a thoughtful expert.",
                max_new_tokens=2048,
            ),
            description="Claude Expert",
        ),
        Discusser(
            model=AsyncAI(
                model_name="gpt-4o",
                system="You are a thoughtful expert.",
            ),
            description="GPT Expert",
        ),
    ]

    # Initialize judge
    judge = AsyncAI(
        model_name="o1-mini",  # Using Claude as judge for its reasoning capabilities
        # system="You are an impartial judge tasked with finding consensus among expert opinions.",
        max_new_tokens=2048,
    )

    # Create consensus AI
    consensus_ai = ConsensusAI(judge=judge, discussers=discussers)

    # Test with a question
    question = "What is the meaning of life?"
    result = asyncio.run(consensus_ai.get_consensus(question))

    print(f"Question: {question}\n")
    print(f"Consensus: {result}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    main()
