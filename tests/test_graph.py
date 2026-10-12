"""Grafo LangGraph, retry, avaliador, fontes, cache e streaming.

LLM falso (sem Ollama) + busca real na Wikipedia, conforme a regra do projeto.
"""
import asyncio
import json

import pytest
from langchain_core.runnables import RunnableLambda

from graph import build_graph
from models.agent import DeepResearchAgent
from schemas.schemas import Avaliacao
from utils.search import CachedSearch, WikipediaSearch


class FakeLLM:
    """Responde conforme o prompt recebido (identifica a chain por uma frase)."""

    def __init__(self, plano, rotas=None, avaliacoes=None, planner_outputs=None):
        self.plano = plano
        self.rotas = rotas or {}
        self.avaliacoes = list(avaliacoes or [{"suficiente": True}])
        self.planner_outputs = list(planner_outputs or [])
        self.calls = {"planner": 0, "router": 0, "evaluator": 0, "answer": 0,
                      "local": 0, "synth": 0}
        self.synth_prompt = ""

    def __call__(self, prompt_value):
        texto = prompt_value.to_string()
        if "planejador de pesquisa" in texto:
            self.calls["planner"] += 1
            if self.planner_outputs:
                return self.planner_outputs.pop(0)
            return json.dumps({"ListOfQueries": [
                {"id": i, "pergunta": p} for i, p in enumerate(self.plano, 1)]})
        if "classifica sub-perguntas" in texto:
            self.calls["router"] += 1
            return json.dumps({"decisoes": [
                {"id": i, "precisa_busca": self.rotas.get(i, True)}
                for i in range(1, 10)]})
        if "avalia se o material" in texto:
            self.calls["evaluator"] += 1
            return json.dumps(self.avaliacoes.pop(0) if len(self.avaliacoes) > 1
                              else self.avaliacoes[0])
        if "nao pesquisavel" in texto:
            self.calls["local"] += 1
            return "resposta local"
        if "SOMENTE os trechos" in texto:
            self.calls["answer"] += 1
            return "resposta com fonte"
        if "redator de pesquisa" in texto:
            self.calls["synth"] += 1
            self.synth_prompt = texto
            return "RELATORIO FINAL"
        raise AssertionError(f"prompt desconhecido: {texto[:80]}")


SHARED_SEARCH = None  # preenchido pela fixture autouse abaixo


@pytest.fixture(autouse=True)
def _use_shared_search(wiki_cached):
    global SHARED_SEARCH
    SHARED_SEARCH = wiki_cached


def make_agent(fake, **kw):
    agent = DeepResearchAgent(
        model_name="fake", search_backend="wikipedia",
        llm=RunnableLambda(fake), **kw)
    agent.search_tool = SHARED_SEARCH
    return agent


async def run_graph(agent, query="Brasil"):
    return await build_graph().ainvoke(
        {"query": query}, config={"configurable": {"agent": agent}})


def test_full_run_with_sources_and_local_route():
    fake = FakeLLM(["Qual é a capital do Brasil?", "Quem é você?"],
                   rotas={2: False})
    agent = make_agent(fake)
    estado = asyncio.run(run_graph(agent))

    assert fake.calls["answer"] == 1 and fake.calls["local"] == 1
    assert fake.calls["evaluator"] == 1
    assert "[sem fonte externa] resposta local" in fake.synth_prompt
    assert estado["report"].startswith("RELATORIO FINAL")
    assert "## Fontes" in estado["report"]
    assert "pt.wikipedia.org" in estado["report"]
    assert {f["id"] for f in estado["findings"]} == {1, 2}


def test_iteration_adds_followup_round():
    fake = FakeLLM(
        ["Qual é a capital do Brasil?"],
        avaliacoes=[
            {"suficiente": False, "novas_perguntas": ["Quando Brasília foi fundada?"]},
            {"suficiente": True},
        ])
    estado = asyncio.run(run_graph(make_agent(fake, max_rounds=3)))

    assert estado["rodada"] == 2
    assert [f["id"] for f in estado["findings"]] == [1, 2]
    assert estado["findings"][1]["pergunta"] == "Quando Brasília foi fundada?"
    assert fake.calls["evaluator"] == 2 and fake.calls["synth"] == 1


def test_iteration_respects_max_rounds():
    fake = FakeLLM(
        ["Qual é a capital do Brasil?"],
        avaliacoes=[{"suficiente": False, "novas_perguntas": ["Outra pergunta sobre Brasília?"]}])
    estado = asyncio.run(run_graph(make_agent(fake, max_rounds=1)))
    assert estado["rodada"] == 1
    assert fake.calls["evaluator"] == 0


def test_followup_skips_repeated_questions():
    agent = make_agent(FakeLLM(["x"]))
    findings = [{"id": 1, "pergunta": "Capital do Brasil?", "resposta": "", "fontes": []}]
    av = Avaliacao(suficiente=False,
                   novas_perguntas=["capital do brasil?", "Nova pergunta?"])
    plano = agent.follow_up_plan(av, findings)
    assert [(s.id, s.pergunta) for s in plano.ListOfQueries] == [(2, "Nova pergunta?")]


def test_planner_retries_on_invalid_json():
    ok = json.dumps({"ListOfQueries": [{"id": 1, "pergunta": "Capital do Brasil?"}]})
    fake = FakeLLM([], planner_outputs=["isto nao e json", ok])
    plano = make_agent(fake, plan_retries=2).plan("Brasil")
    assert fake.calls["planner"] == 2
    assert plano.ListOfQueries[0].pergunta == "Capital do Brasil?"


def test_planner_gives_up_after_retries():
    fake = FakeLLM([], planner_outputs=["lixo", "lixo"])
    with pytest.raises(ValueError):
        make_agent(fake, plan_retries=2).plan("Brasil")
    assert fake.calls["planner"] == 2


def test_evaluator_failure_stops_iteration():
    fake = FakeLLM(["Qual é a capital do Brasil?"], avaliacoes=[{"invalido": 1}])
    estado = asyncio.run(run_graph(make_agent(fake, max_rounds=3)))
    assert estado["rodada"] == 1 and "report" in estado


def test_streaming_emits_tokens_equal_to_report():
    fake = FakeLLM(["Qual é a capital do Brasil?"])
    agent = make_agent(fake)

    async def collect():
        tokens = []
        async for ev in build_graph().astream(
            {"query": "Brasil"}, config={"configurable": {"agent": agent}},
            stream_mode="custom",
        ):
            tokens.append(ev["token"])
        return tokens

    tokens = asyncio.run(collect())
    assert tokens and "".join(tokens).startswith("RELATORIO FINAL")
    assert "## Fontes" in "".join(tokens)


def test_sync_run_wrapper():
    agent = make_agent(FakeLLM(["Qual é a capital do Brasil?"]))
    assert "RELATORIO FINAL" in agent.run("Brasil")


def test_search_cache_hits_on_repeat(wiki_cached):
    cached = CachedSearch(wiki_cached)
    a = cached.search("Machado de Assis")
    b = cached.search("  machado   DE assis ")
    assert a == b and a
    assert (cached.hits, cached.misses) == (1, 1)


def test_search_cache_ttl_expires(wiki_cached):
    cached = CachedSearch(wiki_cached, ttl=0.0)
    cached.search("Machado de Assis")
    cached.search("Machado de Assis")
    assert cached.hits == 0 and cached.misses == 2


def test_default_backend_is_cached():
    agent = make_agent(FakeLLM(["x"]))
    assert isinstance(agent.search_tool, CachedSearch)
