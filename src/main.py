import sys

if sys.version_info < (3, 10):
    sys.exit(
        f"deep-research exige Python 3.10+ (voce esta usando {sys.version.split()[0]}). "
        "Ative o ambiente certo, ex.: conda activate deep-research"
    )

import os
import argparse
import asyncio
import logging

from dotenv import load_dotenv

from models.agent import DeepResearchAgent
from graph import build_graph
from utils.search import SearchError

load_dotenv("./config/.env")


def _env_bool(name, default="false"):
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def parse_args(argv=None):
    """Flags da CLI. Cada uma sobrescreve a variável de ambiente equivalente."""
    p = argparse.ArgumentParser(
        prog="deep-research",
        description="Agente de deep research local (Ollama + LangGraph).")
    p.add_argument("query", nargs="?",
                   help="pergunta única (sem ela, abre o modo interativo)")
    p.add_argument("--model", default=os.getenv("OLLAMA_MODEL_NAME"),
                   help="modelo no Ollama (OLLAMA_MODEL_NAME)")
    p.add_argument("--host", default=os.getenv("OLLAMA_HOST", "http://localhost:11434/"),
                   help="URL do Ollama (OLLAMA_HOST)")
    p.add_argument("--backend", default=os.getenv("SEARCH_BACKEND"),
                   choices=["wikipedia", "duckduckgo", "tavily"],
                   help="backend de busca (SEARCH_BACKEND)")
    p.add_argument("--max-rounds", type=int,
                   default=int(os.getenv("MAX_ROUNDS", "2")),
                   help="máximo de rodadas de pesquisa (MAX_ROUNDS)")
    p.add_argument("--max-subquestions", type=int,
                   default=int(os.getenv("MAX_SUBQUESTIONS", "5")),
                   help="teto de sub-perguntas por rodada (MAX_SUBQUESTIONS)")
    p.add_argument("--temperature", type=float,
                   default=float(os.getenv("TEMPERATURE", "0.1")),
                   help="temperatura do LLM (TEMPERATURE)")
    p.add_argument("--relevance-threshold", type=float,
                   default=float(os.getenv("RELEVANCE_THRESHOLD", "0.4")),
                   help="score mínimo de relevância 0-1 (RELEVANCE_THRESHOLD)")
    p.add_argument("--plan-retries", type=int,
                   default=int(os.getenv("PLAN_RETRIES", "2")),
                   help="tentativas do planner (PLAN_RETRIES)")
    p.add_argument("--wikipedia-lang", default=None,
                   help="idioma da Wikipedia, ex.: pt, en (WIKIPEDIA_LANG)")
    p.add_argument("--thinking", action="store_true",
                   default=_env_bool("enable_thinking"),
                   help="mantém o bloco <think> do modelo (enable_thinking)")
    return p.parse_args(argv)


logging.basicConfig(level=logging.INFO, format="%(message)s")
# O httpx loga cada POST ao Ollama em INFO e polui a saida.
logging.getLogger("httpx").setLevel(logging.WARNING)


async def stream_report(graph, agent, query):
    """Imprime o relatório à medida que o LLM o escreve."""
    print(f"\n{'=' * 70}")
    async for evento in graph.astream(
        {"query": query},
        config={"configurable": {"agent": agent}},
        stream_mode="custom",
    ):
        print(evento.get("token", ""), end="", flush=True)
    print(f"\n{'=' * 70}\n")


def main(argv=None):
    args = parse_args(argv)
    if args.wikipedia_lang:
        os.environ["WIKIPEDIA_LANG"] = args.wikipedia_lang
    try:
        agent = DeepResearchAgent(
            model_name=args.model,
            base_url=args.host,
            temperature=args.temperature,
            enable_thinking=args.thinking,
            max_subquestions=args.max_subquestions,
            search_backend=args.backend,
            relevance_threshold=args.relevance_threshold,
            max_rounds=args.max_rounds,
            plan_retries=args.plan_retries,
        )
    except SearchError as e:
        logging.error("Erro no backend de busca: %s", e)
        return

    graph = build_graph()
    if args.query:
        asyncio.run(stream_report(graph, agent, args.query))
        return

    print("Para encerrar, digite 'exit' ou 'quit'.\n")
    while True:
        try:
            query = input("O que voce quer pesquisar hoje? \n").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAté mais!")
            break

        if query.lower() in ("exit", "quit"):
            print("Até mais!")
            break
        if not query:
            continue

        try:
            asyncio.run(stream_report(graph, agent, query))
        except Exception as e:
            logging.error("Falha ao processar a query: %s", e)
            continue


if __name__ == "__main__":
    main()
