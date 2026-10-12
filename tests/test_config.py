"""Configuração por CLI/env. Os backends são os da Wikipedia (sem rede na construção)."""
from main import parse_args
from utils.search import CachedSearch, get_search_backend


def test_cli_defaults(monkeypatch):
    for v in ("MAX_ROUNDS", "MAX_SUBQUESTIONS", "TEMPERATURE", "SEARCH_BACKEND"):
        monkeypatch.delenv(v, raising=False)
    a = parse_args([])
    assert (a.max_rounds, a.max_subquestions, a.temperature) == (2, 5, 0.1)
    assert a.query is None and a.backend is None


def test_cli_flags_override_env(monkeypatch):
    monkeypatch.setenv("MAX_ROUNDS", "4")
    a = parse_args(["--max-rounds", "1", "--backend", "wikipedia",
                    "--wikipedia-lang", "en", "Brasil"])
    assert a.max_rounds == 1 and a.backend == "wikipedia"
    assert a.wikipedia_lang == "en" and a.query == "Brasil"


def test_env_feeds_defaults(monkeypatch):
    monkeypatch.setenv("MAX_SUBQUESTIONS", "3")
    assert parse_args([]).max_subquestions == 3


def test_wikipedia_lang_from_env(monkeypatch):
    monkeypatch.setenv("WIKIPEDIA_LANG", "en")
    backend = get_search_backend("wikipedia")
    assert backend._backend.lang == "en"
    assert "en.wikipedia.org" in backend._backend.endpoint


def test_cache_ttl_from_env(monkeypatch):
    monkeypatch.setenv("SEARCH_CACHE_TTL", "42")
    backend = get_search_backend("wikipedia")
    assert isinstance(backend, CachedSearch) and backend.ttl == 42.0
