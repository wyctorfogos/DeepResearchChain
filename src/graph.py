import operator
from typing import Annotated, Any, Dict, List, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


class ResearchState(TypedDict, total=False):
    query: str
    plano: Any                     # ListOfQuestions da rodada corrente
    rotas: Dict[int, bool]         # id -> precisa_busca
    findings: Annotated[List[dict], operator.add]  # acumula entre rodadas/ramos
    rodada: int
    continuar: bool                # veredito do avaliador
    report: str


def _agent(config):
    return config["configurable"]["agent"]


async def plan_node(state: ResearchState, config) -> dict:
    plano = await _agent(config).aplan(state["query"])
    return {"plano": plano, "rodada": 1, "findings": []}


async def route_node(state: ResearchState, config) -> dict:
    return {"rotas": await _agent(config).aroute(state["plano"])}


def fan_out(state: ResearchState) -> List[Send]:
    """Uma tarefa `research` por sub-pergunta: o LangGraph as executa em paralelo."""
    return [
        Send("research", {"sub": s, "precisa_busca": state["rotas"][s.id]})
        for s in state["plano"].ListOfQueries
    ]


async def research_node(task: dict, config) -> dict:
    sub = task["sub"]
    resposta, fontes = await _agent(config).aanswer_with_sources(
        sub.pergunta, task["precisa_busca"]
    )
    return {"findings": [{
        "id": sub.id, "pergunta": sub.pergunta,
        "resposta": resposta, "fontes": fontes,
    }]}


async def evaluate_node(state: ResearchState, config) -> dict:
    agent = _agent(config)
    rodada = state.get("rodada", 1)
    if rodada >= agent.max_rounds:
        return {"continuar": False}
    avaliacao = await agent.aevaluate(state["query"], state["findings"])
    if avaliacao.suficiente:
        return {"continuar": False}
    novo = agent.follow_up_plan(avaliacao, state["findings"])
    if not novo.ListOfQueries:
        return {"continuar": False}
    return {"plano": novo, "rodada": rodada + 1, "continuar": True}


def after_evaluate(state: ResearchState) -> str:
    return "route" if state.get("continuar") else "write"


async def write_node(state: ResearchState, config) -> dict:
    """Escreve o relatório emitindo cada pedaço como evento custom (streaming).
    Fora de um `astream`, o writer é um no-op e só o retorno final importa."""
    write = get_stream_writer()
    partes = []
    async for chunk in _agent(config).astream_synthesis(
        state["query"], state["findings"]
    ):
        partes.append(chunk)
        write({"token": chunk})
    return {"report": "".join(partes)}


def build_graph():
    g = StateGraph(ResearchState)
    g.add_node("plan", plan_node)
    g.add_node("route", route_node)
    g.add_node("research", research_node)
    g.add_node("evaluate", evaluate_node)
    g.add_node("write", write_node)

    g.add_edge(START, "plan")
    g.add_edge("plan", "route")
    g.add_conditional_edges("route", fan_out, ["research"])
    g.add_edge("research", "evaluate")
    g.add_conditional_edges("evaluate", after_evaluate, ["route", "write"])
    g.add_edge("write", END)
    return g.compile()
