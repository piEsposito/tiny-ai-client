import asyncio
from typing import List

from pydantic import BaseModel
from structlog import get_logger

from tiny_ai_client import AsyncAI

logger = get_logger(__name__)


class Discusser(BaseModel):
    model: AsyncAI
    description: str | None

    model_config = {"arbitrary_types_allowed": True}


def __str__(self) -> str:
    model_name = self.model.model_name
    desc = f" ({self.description})" if self.description else ""
    return f"Discusser({model_name=}, {desc=})"


class ConsensusAI:
    def __init__(
        self,
        judge: AsyncAI,
        discussers: List[Discusser],
    ):
        self.judge = judge
        self.discussers = discussers

    def build_prompt(
        self, original_prompt: str, responses: List[str], discussers: List[Discusser]
    ) -> str:
        # Prepare formatted responses
        formatted_responses = []
        for response, discusser in zip(responses, discussers):
            desc = discusser.description or "Expert"
            formatted_responses.append(f"{desc}:\n{response}")

        responses_text = "\n\n".join(formatted_responses)

        return f"""
You are the ultimate arbiter of expert analysis. Multiple top experts have responded to the question:
"{original_prompt}"

Below are their responses:
{responses_text}

Your task is to:
1. Summarize each expert’s perspective, including key agreements and disagreements.
2. Thoroughly compare, contrast, and evaluate their ideas, highlighting sources of uncertainty or unresolved points.
3. Provide a cohesive, well-reasoned final answer that incorporates the strongest evidence and arguments from all experts.

Structure your output in three sections:
1. "summary" - a concise overview of the experts’ positions.
2. "reasoning" - a step-by-step process explaining how you weighed each expert’s statements, with references to who said what.
3. "answer" - a singular best-possible conclusion or solution, synthesizing everything you’ve learned.

Adhere to these guidelines for maximum clarity and depth. Strive for nuance where disagreements arise, yet be decisive in recommending a cohesive final stance. Leave no critical point unaddressed.
"""

    async def consult_experts(self, prompt: str, stream: bool = False, **kwargs) -> str:
        # Gather responses from all discussers
        responses = []
        logger.info(
            "Getting responses from discussers",
            prompt=prompt,
            discussers=self.discussers,
            kwargs=kwargs,
        )
        responses = await asyncio.gather(
            *[discusser.model(prompt, **kwargs) for discusser in self.discussers]
        )
        responses = [
            f"Expert {discusser.description}: {response}"
            for discusser, response in zip(self.discussers, responses)
        ]
        for discusser, response in zip(self.discussers, responses):
            logger.info(
                f"Discusser response for {discusser.description}",
                response=response,
            )

        # Format responses for the judge
        judge_prompt = self.build_prompt(prompt, responses, self.discussers)
        return judge_prompt

    async def __call__(self, prompt: str, **kwargs) -> str:
        judge_prompt = await self.consult_experts(prompt, **kwargs)
        return await self.judge(judge_prompt)

    async def astream(self, prompt: str, **kwargs):
        judge_prompt = await self.consult_experts(prompt, **kwargs)
        async for chunk in self.judge.astream(judge_prompt):
            yield chunk


async def main():
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
                model_name="o1-mini",
                max_new_tokens=2048,
            ),
            description="O1 Expert, model great at reasoning",
        ),
    ]

    # Initialize judge
    judge = AsyncAI(
        model_name="claude-3-5-sonnet-20241022",
        system="You are a thoughtful expert.",
        max_new_tokens=2048,
    )

    # Create consensus AI
    consensus_ai = ConsensusAI(judge=judge, discussers=discussers)

    # Test with a question
    question = "What is the best way to learn a new language?"

    async for chunk in consensus_ai.astream(question):
        print(chunk, end="", flush=True)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(main())
