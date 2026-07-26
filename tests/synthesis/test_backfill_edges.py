import asyncio
import json

from s2cs.synthesis.run_backfill_edges import regenerate_pointer_label
from tests.agent.conftest import make_fake_resp


def _resp(obj):
    return make_fake_resp(content=json.dumps(obj))


def test_regenerate_pointer_label(fake_openai_client):
    fake_openai_client.chat.completions.create.return_value = _resp(
        {"pointer_label": "the baseline it compares against"})
    out = asyncio.run(regenerate_pointer_label(
        fake_openai_client, "m",
        citing_evidence="We compare against the prior method [3].",
        to_title="Prior Method", to_abstract="a baseline"))
    assert out == "the baseline it compares against"
