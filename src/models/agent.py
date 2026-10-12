"""
    Agente de Deep Research 
"""
from __future__ import annotations  # PEP 563: permite list[dict] etc. no py3.8

import os
import re
import asyncio
import logging

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from schemas.schemas import ListOfQuestions, RouteDecisions, Avaliacao, SubPerguntas
from utils.search import get_search_backend, SearchError, format_results
from utils.relevance import filter_results
from utils.templates import *

from dotenv import load_dotenv
load_dotenv("./config/.env")

logger = logging.getLogger(__name__)

class DeepResearchAgent:
    """Serviço sem estado. Não sabe nada sobre grafo, loop ou ordem — só executa
    uma das quatro operações quando chamado. Injete uma instância nos nós via
    config["configurable"]["agent"] e os nós ficam funções puras e testáveis."""

    def __init__(
        self,
        model_name: str = None,
        base_url: str = "http://localhost:11434/",
        temperature: float = 0.1,
        enable_thinking: bool = False,
        max_subquestions: int = 5,
        search_backend: str = None,
        relevance_threshold: float = 0.4,
        max_rounds: int = 2,
        plan_retries: int = 2,
        llm=None,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/") + "/v1"
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        self.max_subquestions = max_subquestions
        self.max_rounds = max_rounds
        self.plan_retries = max(1, plan_retries)
        # `llm` injeta um runnable no lugar do ChatOpenAI (testes sem Ollama).
        self._llm_override = llm
        # 0.4 fica no vao medido entre fontes uteis (>=0.50) e ruido (<=0.33).
        self.relevance_threshold = relevance_threshold
        self.search_tool = get_search_backend(search_backend)
        logger.info("[agent] backend de busca ativo: %s", self.search_tool.name)

        # Uma chain por operação. Planner e router ganham um passo extra de
        # validação Pydantic: o LLM apenas sugere, o schema é quem garante.
        self._planner = (
            ChatPromptTemplate.from_template(PLANNER_TEMPLATE)
            | self._llm(json_mode=True)
            | StrOutputParser()
            | RunnableLambda(
                lambda raw: ListOfQuestions.model_validate_json(self._extract_json(raw))
            )
        )
        self._router = (
            ChatPromptTemplate.from_template(ROUTER_TEMPLATE)
            | self._llm(json_mode=True)
            | StrOutputParser()
            | RunnableLambda(
                lambda raw: RouteDecisions.model_validate_json(self._extract_json(raw))
            )
        )
        # Resposta sem fonte: texto livre, sem json_mode.
        self._local = (
            ChatPromptTemplate.from_template(LOCAL_TEMPLATE)
            | self._llm()
            | StrOutputParser()
        )
        self._answerer = (
            ChatPromptTemplate.from_template(ANSWER_TEMPLATE)
            | self._llm()
            | StrOutputParser()
        )
        self._evaluator = (
            ChatPromptTemplate.from_template(EVALUATE_TEMPLATE)
            | self._llm(json_mode=True)
            | StrOutputParser()
            | RunnableLambda(
                lambda raw: Avaliacao.model_validate_json(self._extract_json(raw))
            )
        )
        self._synthesizer = (
            ChatPromptTemplate.from_template(SYNTHESIS_TEMPLATE)
            | self._llm()
            | StrOutputParser()
        )

    # ------------------------------------------------------------------ infra

    def _llm(self, json_mode: bool = False):
        if self._llm_override is not None:
            return self._llm_override
        # Ollama expoe API compativel com OpenAI em /v1; a api_key e ignorada
        # pelo servidor mas exigida pelo cliente.
        llm = ChatOpenAI(
            model=self.model_name,
            base_url=self.base_url,
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
            temperature=self.temperature,
        )
        # json_object e mais confiavel que with_structured_output neste modelo,
        # que anuncia 'tools' mas nao emite tool_calls (structured output volta None).
        return llm.bind(response_format={"type": "json_object"}) if json_mode else llm

    @property
    def _think(self) -> str:
        # Qwen3: desliga o bloco <think>, que suja o JSON do planner/router.
        return "" if self.enable_thinking else " /no_think"

    @staticmethod
    def _extract_json(raw: str) -> str:
        # Rede de seguranca: mesmo em json_mode o modelo as vezes embrulha em
        # ```json ... ``` ou deixa um <think> escapar.
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError(f"Nenhum JSON na resposta do modelo: {raw[:200]}")
        return match.group(0)

    # ------------------------------------------------ operações = nós do grafo

    def plan(self, query: str) -> ListOfQuestions:
        """planner_node: query -> sub-perguntas. Repete se o JSON vier invalido;
        esgotadas as tentativas a exceção propaga — um plano que falha deve
        derrubar a execução, não ser mascarado."""
        entrada = self._plan_input(query)
        for tentativa in range(1, self.plan_retries + 1):
            try:
                plano = self._planner.invoke(entrada)
                break
            except ValueError as e:
                logger.warning("[planner] saida invalida (%d/%d): %s",
                               tentativa, self.plan_retries, e)
                if tentativa == self.plan_retries:
                    raise
        return self._log_plan(plano)

    async def aplan(self, query: str) -> ListOfQuestions:
        entrada = self._plan_input(query)
        for tentativa in range(1, self.plan_retries + 1):
            try:
                plano = await self._planner.ainvoke(entrada)
                break
            except ValueError as e:
                logger.warning("[planner] saida invalida (%d/%d): %s",
                               tentativa, self.plan_retries, e)
                if tentativa == self.plan_retries:
                    raise
        return self._log_plan(plano)

    def _plan_input(self, query: str) -> dict:
        return {
            "query": query,
            "max_subquestions": self.max_subquestions,
            "think_flag": self._think,
        }

    @staticmethod
    def _log_plan(plano: ListOfQuestions) -> ListOfQuestions:
        logger.info("[planner] %d sub-perguntas geradas", len(plano.ListOfQueries))
        for sub in plano.ListOfQueries:
            logger.info("  [%s] %s", sub.id, sub.pergunta)
        return plano

    def route(self, plano: ListOfQuestions) -> dict:
        """
            router_node: decide quais sub-perguntas exigem busca externa.
        """
        try:
            decisoes = self._router.invoke(self._route_input(plano))
        except Exception as e:
            logger.warning("[router] falhou, buscando tudo: %s", e)
            return {s.id: True for s in plano.ListOfQueries}
        return self._apply_route(plano, decisoes)

    async def aroute(self, plano: ListOfQuestions) -> dict:
        try:
            decisoes = await self._router.ainvoke(self._route_input(plano))
        except Exception as e:
            logger.warning("[router] falhou, buscando tudo: %s", e)
            return {s.id: True for s in plano.ListOfQueries}
        return self._apply_route(plano, decisoes)

    def _route_input(self, plano: ListOfQuestions) -> dict:
        perguntas = "\n".join(f"{s.id}. {s.pergunta}" for s in plano.ListOfQueries)
        return {"perguntas": perguntas, "think_flag": self._think}

    @staticmethod
    def _apply_route(plano: ListOfQuestions, decisoes: RouteDecisions) -> dict:
        # Casa por id, nao por posicao: o modelo pode devolver fora de ordem,
        # repetido ou incompleto. Ids ausentes ficam True por padrao.
        mapa = {s.id: True for s in plano.ListOfQueries}
        for d in decisoes.decisoes:
            if d.id in mapa:
                mapa[d.id] = d.precisa_busca
        for s in plano.ListOfQueries:
            if not mapa[s.id]:
                logger.info("[router] [%s] sem busca: %s", s.id, s.pergunta)
        return mapa

    # ------------------------------------------------------------ resposta

    def _retrieve(self, pergunta: str):
        """Busca + filtro de relevância. Devolve (relevantes, erro): exatamente um
        dos dois é não-nulo. `relevantes` = [(resultado, score)]."""
        try:
            resultados = self.search_tool.search(pergunta)
        except SearchError as e:
            logger.warning("[answer] busca falhou: %s", e)
            return None, f"[nao foi possivel apurar: {e}]"
        if not resultados:
            logger.info("[answer] busca sem resultados p/: %s", pergunta)
            return None, "[nao foi possivel apurar: busca sem resultados]"

        relevantes = filter_results(pergunta, resultados, self.relevance_threshold)
        if not relevantes:
            melhor = max(
                (s for _, s in filter_results(pergunta, resultados, 0.0)),
                default=0.0,
            )
            logger.info(
                "[answer] %s: %d resultados, nenhum relevante "
                "(melhor score %.2f < %.2f)",
                self.search_tool.name, len(resultados), melhor,
                self.relevance_threshold,
            )
            return None, "[nao foi possivel apurar: nenhuma fonte relevante]"

        logger.info("[answer] %s: %d/%d fontes relevantes",
                    self.search_tool.name, len(relevantes), len(resultados))
        return relevantes, None

    @staticmethod
    def _sources(relevantes) -> list[dict]:
        return [{"title": r["title"], "url": r["url"]} for r, _ in relevantes]

    def answer_with_sources(self, pergunta: str, precisa_busca: bool = True):
        """pesquisador_node: uma sub-pergunta -> (resposta, fontes)."""
        if not precisa_busca:
            logger.info("[answer] resposta local (nao pesquisavel): %s", pergunta)
            try:
                resposta = self._local.invoke(
                    {"pergunta": pergunta, "think_flag": self._think})
                return f"{SEM_FONTE} {resposta}", []
            except Exception as e:
                logger.warning("[answer] LLM local falhou: %s", e)
                return f"[nao foi possivel apurar: {e}]", []

        relevantes, erro = self._retrieve(pergunta)
        if erro:
            return erro, []
        try:
            resposta = self._answerer.invoke({
                "evidencia": format_results([r for r, _ in relevantes]),
                "pergunta": pergunta,
                "think_flag": self._think,
            })
        except Exception as e:
            logger.warning("[answer] LLM falhou: %s", e)
            return f"[nao foi possivel apurar: {e}]", []
        return resposta, self._sources(relevantes)

    async def aanswer_with_sources(self, pergunta: str, precisa_busca: bool = True):
        if not precisa_busca:
            logger.info("[answer] resposta local (nao pesquisavel): %s", pergunta)
            try:
                resposta = await self._local.ainvoke(
                    {"pergunta": pergunta, "think_flag": self._think})
                return f"{SEM_FONTE} {resposta}", []
            except Exception as e:
                logger.warning("[answer] LLM local falhou: %s", e)
                return f"[nao foi possivel apurar: {e}]", []

        # A busca usa `requests` (bloqueante): roda em thread para as
        # sub-perguntas realmente andarem em paralelo.
        relevantes, erro = await asyncio.to_thread(self._retrieve, pergunta)
        if erro:
            return erro, []
        try:
            resposta = await self._answerer.ainvoke({
                "evidencia": format_results([r for r, _ in relevantes]),
                "pergunta": pergunta,
                "think_flag": self._think,
            })
        except Exception as e:
            logger.warning("[answer] LLM falhou: %s", e)
            return f"[nao foi possivel apurar: {e}]", []
        return resposta, self._sources(relevantes)

    def answer(self, pergunta: str, precisa_busca: bool = True) -> str:
        return self.answer_with_sources(pergunta, precisa_busca)[0]

    # ------------------------------------------------------------ avaliação

    async def aevaluate(self, query: str, findings: list[dict]) -> Avaliacao:
        """avaliador_node: o apurado basta? Se a avaliação falhar, considera
        suficiente — melhor entregar o que há do que travar o loop."""
        try:
            return await self._evaluator.ainvoke({
                "query": query,
                "findings": self._findings_blob(findings),
                "max_novas": self.max_subquestions,
                "think_flag": self._think,
            })
        except Exception as e:
            logger.warning("[avaliador] falhou, encerrando iteração: %s", e)
            return Avaliacao(suficiente=True, motivo=f"avaliacao falhou: {e}")

    def follow_up_plan(self, avaliacao: Avaliacao, findings: list[dict]) -> ListOfQuestions:
        """Transforma as novas perguntas do avaliador em um plano com ids que
        continuam a numeração, sem repetir perguntas já feitas."""
        feitas = {f["pergunta"].strip().lower() for f in findings}
        proximo = max((f["id"] for f in findings), default=0) + 1
        subs = []
        for texto in avaliacao.novas_perguntas[: self.max_subquestions]:
            if texto.strip().lower() in feitas:
                continue
            subs.append(SubPerguntas(id=proximo, pergunta=texto.strip()))
            feitas.add(texto.strip().lower())
            proximo += 1
        return ListOfQuestions(ListOfQueries=subs)

    # ------------------------------------------------------------ síntese

    @staticmethod
    def _findings_blob(findings: list[dict]) -> str:
        ordenados = sorted(findings, key=lambda f: f["id"])
        return "\n\n".join(
            f'{f["id"]}. {f["pergunta"]}\n{f["resposta"]}' for f in ordenados
        )

    @staticmethod
    def format_sources(findings: list[dict]) -> str:
        """Seção 'Fontes' montada em código (não pelo LLM): só URLs realmente
        recuperadas, sem duplicatas, na ordem em que aparecem."""
        vistas, linhas = set(), []
        for f in sorted(findings, key=lambda f: f["id"]):
            for fonte in f.get("fontes", []):
                if fonte["url"] in vistas:
                    continue
                vistas.add(fonte["url"])
                linhas.append(f'- [{fonte["title"]}]({fonte["url"]})')
        return "## Fontes\n" + "\n".join(linhas) if linhas else ""

    def _synth_input(self, query: str, findings: list[dict]) -> dict:
        return {
            "query": query,
            "findings": self._findings_blob(findings),
            "think_flag": self._think,
        }

    def synthesize(self, query: str, findings: list[dict]) -> str:
        """escritor_node: findings [{id, pergunta, resposta, fontes?}] -> relatório."""
        corpo = self._synthesizer.invoke(self._synth_input(query, findings))
        fontes = self.format_sources(findings)
        return f"{corpo}\n\n{fontes}" if fontes else corpo

    async def astream_synthesis(self, query: str, findings: list[dict]):
        """Gera o relatório em pedaços (streaming); a seção Fontes vem por último."""
        async for chunk in self._synthesizer.astream(self._synth_input(query, findings)):
            yield chunk
        fontes = self.format_sources(findings)
        if fontes:
            yield f"\n\n{fontes}"

    # ------------------------------------------------------------ execução

    async def arun(self, query: str) -> str:
        from graph import build_graph  # import tardio: evita ciclo
        estado = await build_graph().ainvoke(
            {"query": query},
            config={"configurable": {"agent": self}},
        )
        return estado["report"]

    def run(self, query: str) -> str:
        return asyncio.run(self.arun(query))
