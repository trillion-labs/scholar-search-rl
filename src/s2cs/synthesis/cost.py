from types import SimpleNamespace
from typing import Any

import openai


def _usage_cost(usage: Any) -> float | None:
    if usage is None:
        return None
    cost = getattr(usage, "cost", None)
    if cost is None and hasattr(usage, "model_dump"):
        try:
            cost = usage.model_dump().get("cost")
        except Exception:
            cost = None
    return float(cost) if cost is not None else None


class CostTrackingClient:
    """Wrap an AsyncOpenAI client to sum OpenRouter-reported `usage.cost`.

    Only the `chat.completions.create` path is proxied — the single method our
    LLM helpers call — so synthesis (and the judge it drives) can account for
    spend without baking cost state into `s2cs.agent.llm`, which the agent and
    trainer paths also use. Local servers (sglang/vLLM) report no cost, so
    `total_cost` and `calls_with_cost` stay 0.
    """

    def __init__(self, client: openai.AsyncOpenAI):
        self._client = client
        self.total_cost = 0.0
        self.calls = 0
        self.calls_with_cost = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any):
        resp = await self._client.chat.completions.create(**kwargs)
        self.calls += 1
        cost = _usage_cost(getattr(resp, "usage", None))
        if cost is not None:
            self.total_cost += cost
            self.calls_with_cost += 1
        return resp
