from __future__ import annotations  
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

from dotenv import load_dotenv
load_dotenv("./config/.env")

logger = logging.getLogger(__name__)


PLANNER_TEMPLATE = """Voce e um planejador de pesquisa. Decomponha a pergunta do usuario em ate \
{max_subquestions} sub-perguntas objetivas, autocontidas e pesquisaveis na web.
Cada sub-pergunta deve cobrir um aspecto diferente, sem repetir as outras.

{max_subquestions} e um TETO, nao uma meta. Uma pergunta simples e direta deve gerar \
UMA unica sub-pergunta. So decomponha em varias quando a pergunta realmente tiver \
aspectos independentes que exigem apuracao separada.

Responda SOMENTE com JSON valido, sem nenhum texto ao redor, no formato:
{{"ListOfQueries": [{{"id": 1, "pergunta": "..."}}]}}

Pergunta: {query}{think_flag}"""

ROUTER_TEMPLATE = """Voce classifica sub-perguntas de pesquisa. Para cada uma, decida se \
responde-la exige consultar fontes externas (web).

Marque "precisa_busca": false APENAS quando NENHUMA fonte externa poderia responder:
- perguntas sobre o proprio assistente (seu nome, o que voce e)
- calculo puro ou reformatacao de algo ja dado na pergunta
- pedidos de opiniao ou preferencia

Marque "precisa_busca": true para TODO o resto, incluindo fatos que voce acredita saber \
de cabeca - o objetivo do sistema e responder com fonte citavel, nao de memoria.
Na duvida, use true.

Responda SOMENTE com JSON valido, sem texto ao redor:
{{"decisoes": [{{"id": 1, "precisa_busca": true, "motivo": "..."}}]}}

Sub-perguntas:
{perguntas}{think_flag}"""

ANSWER_TEMPLATE = """Responda a pergunta usando SOMENTE os trechos abaixo.
Nao complete com conhecimento proprio. Se os trechos nao respondem a pergunta, \
diga exatamente: NAO ENCONTRADO NAS FONTES.
Cite o titulo da fonte entre colchetes ao afirmar cada fato.

Trechos:
{evidencia}

Pergunta: {pergunta}{think_flag}"""

SYNTHESIS_TEMPLATE = """Voce e um redator de pesquisa. Abaixo estao sub-perguntas e o que \
foi apurado para cada uma.

Escreva uma resposta unica e coerente para a pergunta original, usando SOMENTE o que \
foi apurado. Se algo nao foi apurado, diga explicitamente que nao foi possivel verificar \
em vez de completar com conhecimento proprio.

Pergunta original: {query}

Apurado:
{findings}{think_flag}"""

# Sentinel: string que a sintese deve tratar como "nao verificado".
NAO_APLICAVEL = "[nao aplicavel: pergunta nao depende de fontes externas]"


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
        """router_node: decide quais sub-perguntas exigem busca externa.

        Classifica o lote inteiro em UMA chamada de LLM, não uma por pergunta.
        Retorna {id: bool}.

        FAIL-OPEN: qualquer falha (LLM fora, JSON inválido, id faltando)
        resolve para True. Num sistema de pesquisa o erro barato é buscar a
        mais; o erro caro é responder sem fonte porque o classificador quebrou
        em silêncio.
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
        """pesquisador_node: uma sub-pergunta -> resposta ancorada nas fontes.

        NUNCA lança: uma sub-pergunta que falha vira um sentinel que a síntese
        trata como 'não verificado'. É isso que deixa o fan-out paralelo seguro —
        um worker que estoura não pode derrubar os outros nem o grafo inteiro.

        precisa_busca=False NÃO libera resposta de memória: devolve sentinel e
        deixa a síntese tratar. Toda a garantia de citação depende disso.
        """
        if not precisa_busca:
            logger.info("[answer] sem busca (nao pesquisavel): %s", pergunta)
            return NAO_APLICAVEL

        try:
            resultados = self.search_tool.search(pergunta)
        except SearchError as e:
            logger.warning("[answer] busca falhou: %s", e)
            return f"[nao foi possivel apurar: {e}]"
        if not resultados:
            logger.info("[answer] busca sem resultados p/: %s", pergunta)
            return "[nao foi possivel apurar: busca sem resultados]"

        # Descartar o irrelevante aqui evita gastar uma chamada de LLM para ler
        # texto que nao responde a pergunta.
        relevantes = filter_results(pergunta, resultados, self.relevance_threshold)
        if not relevantes:
            # Re-pontua sem corte para saber o quao perto ficou do threshold —
            # e isso que distingue 'filtro agressivo demais' de 'fonte ruim'.
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
        """escritor_node: findings [{id, pergunta, resposta}] -> relatório único.

        Ordena por id antes de montar o texto. A ordem de chegada do fan-out
        paralelo NÃO é garantida, então nunca confie na posição da lista
        (era o que o zip() da versão antiga fazia — quebra assim que paraleliza).
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

    # ------------------------------------------------------------- transitório

    def run(self, query: str) -> str:
        """Orquestração sequencial mínima, só para testar as operações antes de o
        grafo existir. Some quando o StateGraph assumir — o for-loop aqui é
        exatamente o que o fan-out + reflexão vão substituir."""
        plano = self.plan(query)
        rotas = self.route(plano)
        findings = [
            {"id": s.id, "pergunta": s.pergunta,
             "resposta": self.answer(s.pergunta, rotas[s.id])}
            for s in plano.ListOfQueries
        ]
        return self.synthesize(query, findings)