import os
import logging

from dotenv import load_dotenv

from models.agent import DeepResearchAgent
from utils.search import SearchError

load_dotenv("./config/.env")

OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434/")
ENABLE_THINKING = os.getenv("enable_thinking", "false").strip().lower() in (
    "1", "true", "yes", "on"
)
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.4"))

logging.basicConfig(level=logging.INFO, format="%(message)s")
# O httpx loga cada POST ao Ollama em INFO e polui a saida.
logging.getLogger("httpx").setLevel(logging.WARNING)


def main():
    try:
        agent = DeepResearchAgent(
            model_name=OLLAMA_MODEL_NAME,
            base_url=OLLAMA_HOST,
            enable_thinking=ENABLE_THINKING,
            relevance_threshold=RELEVANCE_THRESHOLD,
        )
    except SearchError as e:
        logging.error("Erro no backend de busca: %s", e)
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
            resposta = agent.run(query)
        except Exception as e:
            logging.error("Falha ao processar a query: %s", e)
            continue

        print(f"\n{'=' * 70}\n{resposta}\n{'=' * 70}\n")


if __name__ == "__main__":
    main()