from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def fake_openai_client():
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock()
    return client
