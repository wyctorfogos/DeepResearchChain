"""
    Agente de Deep Research 
"""
from __future__ import annotations  # PEP 563: permite list[dict] etc. no py3.8

import os
import re
import logging

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from schemas.schemas import ListOfQuestions, RouteDecisions
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
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/") + "/v1"
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        self.max_subquestions = max_subquestions
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
        self._synthesizer = (
            ChatPromptTemplate.from_template(SYNTHESIS_TEMPLATE)
            | self._llm()
            | StrOutputParser()
        )

    # ------------------------------------------------------------------ infra

    def _llm(self, json_mode: bool = False):
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
        """planner_node: query -> sub-perguntas. Deixa exceção propagar de
        propósito — um plano que falha deve derrubar a execução, não ser mascarado."""
        plano = self._planner.invoke({
            "query": query,
            "max_subquestions": self.max_subquestions,
            "think_flag": self._think,
        })
        logger.info("[planner] %d sub-perguntas geradas", len(plano.ListOfQueries))
        for sub in plano.ListOfQueries:
            logger.info("  [%s] %s", sub.id, sub.pergunta)
        return plano

    def route(self, plano: ListOfQuestions) -> dict:
        """
            router_node: decide quais sub-perguntas exigem busca externa.
        """
        todos_true = {s.id: True for s in plano.ListOfQueries}
        perguntas = "\n".join(f"{s.id}. {s.pergunta}" for s in plano.ListOfQueries)
        try:
            decisoes = self._router.invoke({
                "perguntas": perguntas,
                "think_flag": self._think,
            })
        except Exception as e:
            logger.warning("[router] falhou, buscando tudo: %s", e)
            return todos_true

        # Casa por id, nao por posicao: o modelo pode devolver fora de ordem,
        # repetido ou incompleto. Ids ausentes ficam True pelo default acima.
        mapa = dict(todos_true)
        for d in decisoes.decisoes:
            if d.id in mapa:
                mapa[d.id] = d.precisa_busca

        for s in plano.ListOfQueries:
            if not mapa[s.id]:
                logger.info("[router] [%s] sem busca: %s", s.id, s.pergunta)
        return mapa

    def answer(self, pergunta: str, precisa_busca: bool = True) -> str:
        """
            pesquisador_node: uma sub-pergunta -> resposta.
        """
        if not precisa_busca:
            logger.info("[answer] resposta local (nao pesquisavel): %s", pergunta)
            try:
                resposta = self._local.invoke({
                    "pergunta": pergunta,
                    "think_flag": self._think,
                })
                return f"{SEM_FONTE} {resposta}"
            except Exception as e:
                logger.warning("[answer] LLM local falhou: %s", e)
                return f"[nao foi possivel apurar: {e}]"

        try:
            resultados = self.search_tool.search(pergunta)
        except SearchError as e:
            logger.warning("[answer] busca falhou: %s", e)
            return f"[nao foi possivel apurar: {e}]"
        if not resultados:
            logger.info("[answer] busca sem resultados p/: %s", pergunta)
            return "[nao foi possivel apurar: busca sem resultados]"

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
            return "[nao foi possivel apurar: nenhuma fonte relevante]"

        evidencia = format_results([r for r, _ in relevantes])
        logger.info(
            "[answer] %s: %d/%d fontes relevantes, %d chars de evidencia",
            self.search_tool.name, len(relevantes), len(resultados), len(evidencia),
        )
        try:
            return self._answerer.invoke({
                "evidencia": evidencia,
                "pergunta": pergunta,
                "think_flag": self._think,
            })
        except Exception as e:
            logger.warning("[answer] LLM falhou: %s", e)
            return f"[nao foi possivel apurar: {e}]"

    def synthesize(self, query: str, findings: list[dict]) -> str:
        """
            escritor_node: findings [{id, pergunta, resposta}] -> relatório único.
        """
        ordenados = sorted(findings, key=lambda f: f["id"])
        blob = "\n\n".join(
            f'{f["id"]}. {f["pergunta"]}\n{f["resposta"]}' for f in ordenados
        )
        return self._synthesizer.invoke({
            "query": query,
            "findings": blob,
            "think_flag": self._think,
        })

    def run(self, query: str) -> str:
        plano = self.plan(query)
        rotas = self.route(plano)
        findings = [
            {"id": s.id, "pergunta": s.pergunta,
             "resposta": self.answer(s.pergunta, rotas[s.id])}
            for s in plano.ListOfQueries
        ]
        return self.synthesize(query, findings)