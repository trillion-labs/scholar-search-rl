import asyncio

import httpx
import openai

from s2cs.agent.judge import judge
from tests.agent.conftest import make_fake_resp


def test_judge_correct(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = make_fake_resp(
        content='{"reasoning": "same year", "judgment": "Correct"}'
    )
    v = asyncio.run(judge("Q?", "2017", "the year was 2017", client=fake_openai_client, model="m"))
    assert v.verdict == "Correct"
    assert "same year" in v.reasoning


def test_judge_incorrect(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = make_fake_resp(
        content='{"reasoning": "different", "judgment": "Incorrect"}'
    )
    v = asyncio.run(judge("Q?", "2017", "1995", client=fake_openai_client, model="m"))
    assert v.verdict == "Incorrect"


def test_judge_falls_back_to_incorrect_on_json_failure(fake_openai_client):
    fake_openai_client.chat.completions.create.side_effect = [
        make_fake_resp(content="not json 1"),
        make_fake_resp(content="not json 2"),
        make_fake_resp(content="not json 3"),
    ]
    v = asyncio.run(judge("Q?", "x", "y", client=fake_openai_client, model="m"))
    assert v.verdict == "Incorrect"
    assert "parse" in v.reasoning.lower()


def test_judge_normalizes_unexpected_verdict_to_incorrect(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = make_fake_resp(
        content='{"reasoning": "huh", "judgment": "maybe"}'
    )
    v = asyncio.run(judge("Q?", "x", "y", client=fake_openai_client, model="m"))
    assert v.verdict == "Incorrect"


def test_judge_falls_back_to_incorrect_on_api_error(fake_openai_client):
    fake_openai_client.chat.completions.create.side_effect = openai.APIError(
        "rate limited", request=httpx.Request("POST", "http://x"), body=None
    )
    v = asyncio.run(judge("Q?", "x", "y", client=fake_openai_client, model="m"))
    assert v.verdict == "Incorrect"
    assert "error" in v.reasoning.lower()
