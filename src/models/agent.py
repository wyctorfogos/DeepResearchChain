import os
import re
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from schemas.schemas import InputState, ListOfQuestions
from utils.search import get_search_backend, SearchError

from dotenv import load_dotenv
load_dotenv("./config/.env")

# As chaves do JSON de exemplo sao escapadas ({{ }}) porque o ChatPromptTemplate
# trata { } como variavel de template.
PLANNER_TEMPLATE = """Voce e um planejador de pesquisa. Decomponha a pergunta do usuario em ate \
{max_subquestions} sub-perguntas objetivas, autocontidas e pesquisaveis na web.
Cada sub-pergunta deve cobrir um aspecto diferente, sem repetir as outras.

Responda SOMENTE com JSON valido, sem nenhum texto ao redor, no formato:
{{"ListOfQueries": [{{"id": 1, "pergunta": "..."}}]}}

Pergunta: {query}{think_flag}"""

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


class DeepResearchAgent:
    def __init__(self,
            model_name:str=None,
            ipaddress_hostname:str="http://localhost:11434/",
            temperature:float=0.1,
            enable_thinking:bool=False,
            max_subquestions:int=5,
            search_backend:str=None,
            use_agent:bool=False
        ):
        self.model_name=model_name
        self.ipaddress_hostname=ipaddress_hostname
        self.temperature=temperature
        self.enable_thinking=enable_thinking
        self.max_subquestions=max_subquestions
        # use_agent=True devolve a decisao de buscar ao LLM. Neste modelo isso
        # significa nao buscar nunca (0 tool calls medidas), entao o padrao e a
        # recuperacao deterministica: o codigo busca, o LLM so resume.
        self.use_agent=use_agent

        self.search_tool = get_search_backend(search_backend)

        # Chains sem tools: planejar, responder ancorado, redigir.
        self.planner_chain = self.build_planner_chain()
        self.answer_chain = self.build_answer_chain()
        self.synthesis_chain = self.build_synthesis_chain()
        self.agente = self.initialize_agent_model() if use_agent else None

    # ------------------------------------------------------------------ infra

    def build_llm(self, json_mode=False):
        # O Ollama expoe uma API compativel com a da OpenAI em /v1.
        # A api_key e ignorada pelo servidor, mas o cliente exige um valor.
        llm = ChatOpenAI(
            model=self.model_name,
            base_url=self.ipaddress_hostname.rstrip("/") + "/v1",
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
            temperature=self.temperature,
        )
        if json_mode:
            # Mais confiavel que with_structured_output: este modelo anuncia
            # 'tools' mas nao emite tool_calls, e o structured output volta None.
            return llm.bind(response_format={"type": "json_object"})
        return llm

    @property
    def think_flag(self):
        # Qwen3 desliga o bloco <think>, que suja o JSON e o parser do agente.
        return "" if self.enable_thinking else " /no_think"

    @staticmethod
    def extract_json(raw):
        # O modelo pode envolver o JSON em ```json ... ``` ou num bloco <think>.
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError(f"Nenhum JSON na resposta do modelo: {raw[:200]}")
        return match.group(0)

    # ----------------------------------------------------------------- chains

    def build_planner_chain(self):
        # prompt | llm | str | validacao Pydantic. A validacao e o que garante o
        # schema - o LLM apenas sugere.
        parse = RunnableLambda(
            lambda raw: ListOfQuestions.model_validate_json(self.extract_json(raw))
        )
        return (
            ChatPromptTemplate.from_template(PLANNER_TEMPLATE)
            | self.build_llm(json_mode=True)
            | StrOutputParser()
            | parse
        )

    def build_answer_chain(self):
        return (
            ChatPromptTemplate.from_template(ANSWER_TEMPLATE)
            | self.build_llm()
            | StrOutputParser()
        )

    def build_synthesis_chain(self):
        return (
            ChatPromptTemplate.from_template(SYNTHESIS_TEMPLATE)
            | self.build_llm()
            | StrOutputParser()
        )

    def initialize_agent_model(self):
        try:
            # O agente usa o mesmo backend plugavel, embrulhado como Tool.
            search_tool = Tool(
                name="web_search",
                func=self.search_tool.run,
                description="Busca informacao factual na web. Entrada: uma pergunta.",
            )

            # OPENAI_FUNCTIONS usa tool calling nativo. O ReAct textual
            # (ZERO_SHOT_REACT_DESCRIPTION) faz o modelo 4B girar ate o limite
            # de iteracoes porque ele nao emite "Final Answer:".
            agent = initialize_agent(
                tools=[search_tool],
                llm=self.build_llm(),
                agent=AgentType.OPENAI_FUNCTIONS,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5,
            )
            return agent
        except Exception as e:
            raise ValueError(f"Error ao instanciar o agente:{e}\n")

    # ------------------------------------------------------------- pipeline

    def break_query_into_subquestions(self, query):
        try:
            return self.planner_chain.invoke({
                "query": query,
                "max_subquestions": self.max_subquestions,
                "think_flag": self.think_flag,
            })
        except Exception as e:
            raise ValueError(f"Erro ao quebrar a query em sub-perguntas: {e}\n")

    def answer_subquestion(self, subpergunta):
        # Uma sub-pergunta que falha nao derruba a pesquisa inteira - a etapa de
        # sintese e instruida a tratar isso como "nao verificado".
        if self.use_agent:
            try:
                return self.agente.run(subpergunta.pergunta + self.think_flag)
            except Exception as e:
                return f"[nao foi possivel apurar: {e}]"

        # Quem decide buscar e o codigo, nao o LLM - e so assim a evidencia
        # entra de fato no contexto.
        try:
            evidencia = self.search_tool.run(subpergunta.pergunta)
        except SearchError as e:
            return f"[nao foi possivel apurar: {e}]"
        if not evidencia.strip():
            return "[nao foi possivel apurar: busca sem resultados]"

        print(f"    ({self.search_tool.name}: {len(evidencia)} chars recuperados)")
        try:
            return self.answer_chain.invoke({
                "evidencia": evidencia,
                "pergunta": subpergunta.pergunta,
                "think_flag": self.think_flag,
            })
        except Exception as e:
            return f"[nao foi possivel apurar: {e}]"

    def synthesize(self, query, plano, respostas):
        findings = "\n\n".join(
            f"{sub.id}. {sub.pergunta}\n{resposta}"
            for sub, resposta in zip(plano.ListOfQueries, respostas)
        )
        try:
            return self.synthesis_chain.invoke({
                "query": query,
                "findings": findings,
                "think_flag": self.think_flag,
            })
        except Exception as e:
            raise ValueError(f"Erro ao sintetizar a resposta: {e}\n")

    def request_query(self, query):
        try:
            # Usar o Pydantic para verificar o filtro
            filtered_query = InputState(input_query=query)
            pergunta = filtered_query.input_query

            plano = self.break_query_into_subquestions(pergunta)
            print(f"\n[planner] {len(plano.ListOfQueries)} sub-perguntas:")
            for sub in plano.ListOfQueries:
                print(f"  [{sub.id}] {sub.pergunta}")

            respostas = []
            for sub in plano.ListOfQueries:
                print(f"\n[agente] respondendo [{sub.id}] {sub.pergunta}")
                respostas.append(self.answer_subquestion(sub))

            print("\n[sintese] redigindo resposta final...")
            return self.synthesize(pergunta, plano, respostas)
        except Exception as e:
            raise ValueError(f"Erro ao realizar a query: {e}\n")
