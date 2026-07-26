import asyncio

from s2cs.agent.llm import chat_json
from s2cs.synthesis.cost import CostTrackingClient
from tests.agent.conftest import make_fake_resp


def test_tracks_cost_per_call(fake_openai_client):
    fake_openai_client.chat.completions.create.side_effect = [
        make_fake_resp(content="a", cost=0.001),
        make_fake_resp(content="b", cost=None),
    ]
    client = CostTrackingClient(fake_openai_client)

    async def run():
        await client.chat.completions.create(model="m", messages=[])
        await client.chat.completions.create(model="m", messages=[])

    asyncio.run(run())
    assert abs(client.total_cost - 0.001) < 1e-9
    assert client.calls == 2
    assert client.calls_with_cost == 1


def test_tracks_cost_through_chat_json(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = make_fake_resp(content='{"k": 1}', cost=0.002)
    client = CostTrackingClient(fake_openai_client)
    out = asyncio.run(chat_json(client, "m", [{"role": "user", "content": "x"}]))
    assert out == {"k": 1}
    assert abs(client.total_cost - 0.002) < 1e-9
    assert client.calls_with_cost == 1
