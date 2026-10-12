"""Testes de integracao: todos usam a Wikipedia real (regra do projeto)."""
import pytest

from models.agent import DeepResearchAgent
from schemas.schemas import ListOfQuestions
from utils.relevance import content_terms, filter_results, score
from utils.search import SearchError, WikipediaSearch, get_search_backend


@pytest.fixture(scope="module")
def wiki(wiki_cached):
    return wiki_cached


@pytest.fixture(scope="module")
def brasil(wiki):
    return wiki.search("Brasil capital Brasília")


def test_search_returns_structured_results(brasil):
    assert brasil
    for r in brasil:
        assert set(r) == {"title", "url", "text"}
        assert r["url"].startswith("https://pt.wikipedia.org/")
        assert 0 < len(r["text"]) <= 1200


def test_run_formats_text(wiki):
    out = wiki.run("Machado de Assis")
    assert "[" in out and "wikipedia.org" in out


def test_no_results_returns_empty(wiki):
    assert wiki.search("zxqvjkwpqz xkcdqqzz") == []


def test_unreachable_endpoint_raises_search_error():
    w = WikipediaSearch(lang="invalid-lang-xyz", timeout=3)
    with pytest.raises(SearchError):
        w.search("teste")


def test_backend_factory():
    assert get_search_backend("wikipedia").name == "wikipedia"
    with pytest.raises(SearchError):
        get_search_backend("inexistente")


def test_relevance_scores_real_results(brasil):
    q = "Qual é a capital do Brasil?"
    kept = filter_results(q, brasil, 0.0)
    scores = [s for _, s in kept]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert max(scores) >= 0.4


def test_relevance_rejects_unrelated_query(brasil):
    assert filter_results("fisica quantica de buracos negros", brasil, 0.4) == []


def test_content_terms_strips_accents_and_stopwords():
    # "são" é stopword; os demais termos saem sem acento e em radical
    assert content_terms("A Vitória do São Paulo") == {"vitori", "paul"}


def test_score_empty_query_is_zero(brasil):
    assert score("a de o", brasil[0]) == 0.0


def test_agent_reports_when_no_relevant_source():
    agent = DeepResearchAgent(model_name="x", search_backend="wikipedia")
    # Falha antes de qualquer chamada ao LLM: nao precisa de Ollama.
    resp = agent.answer("zxqvjkwpqz xkcdqqzz")
    assert resp.startswith("[nao foi possivel apurar")


def test_json_extraction_and_schema():
    raw = '```json\n{"ListOfQueries": [{"id": 1, "pergunta": "x?"}]}\n```'
    plano = ListOfQuestions.model_validate_json(DeepResearchAgent._extract_json(raw))
    assert plano.ListOfQueries[0].id == 1
    with pytest.raises(ValueError):
        DeepResearchAgent._extract_json("sem json")


def test_stem_matches_inflections():
    from utils.relevance import stem
    assert stem("escritos") == stem("escrito")
    assert stem("fundacao") == stem("fundacoes")
    assert stem("printing") == stem("printed")
    assert stem("fundada") == stem("fundado") == stem("fundacao")
    assert stem("sol") == "sol" and stem("mes") == "mes"  # curtas intactas


def test_stemming_improves_recall_on_real_results(wiki):
    q = "Quando Brasília foi fundada?"
    resultados = wiki.search("Brasília")
    best = max(score(q, r) for r in resultados)
    assert best >= 0.4
