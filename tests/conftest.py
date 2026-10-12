import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


import pytest

from utils.search import CachedSearch, WikipediaSearch


@pytest.fixture(scope="session")
def wiki_cached():
    """Um unico backend Wikipedia com cache para toda a sessao: poupa a API
    (rate limit 429) sem deixar de usar a Wikipedia real."""
    return CachedSearch(WikipediaSearch())
