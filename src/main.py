import os
from dotenv import load_dotenv
from models.agent import DeepResearchAgent
import logging

# Carregar os dados 
load_dotenv("./config/.env")
OLLAMA_MODEL_NAME=os.getenv("OLLAMA_MODEL_NAME")
OLLAMA_HOST=os.getenv("OLLAMA_HOST")
enable_thinking=os.getenv("enable_thinking", "false").strip().lower() in ("1", "true", "yes", "on")

if __name__=="__main__":
    deep_search_agent = DeepResearchAgent(model_name=OLLAMA_MODEL_NAME, ipaddress_hostname=OLLAMA_HOST, enable_thinking=enable_thinking)
    logging.info("To finish the session, just digit 'exit' or 'quit'!")
    while True:
        # Get the user input
        input_text=input("What do you want to search today? ").strip()
        if input_text in ["exit", "quit"]:
            logging.info(f"Bye bye!")
            break
        response = deep_search_agent.request_query(query=input_text)
        print(f"Response:{response}\n")

    