# DeepResearchChain

Agente de *deep research* em linha de comando. Ele recebe uma pergunta ampla, a decompõe em sub-perguntas, busca evidências na web para cada uma, responde **apenas com base nas fontes encontradas** e, no fim, sintetiza tudo em um relatório único.

O LLM roda localmente via [Ollama](https://ollama.com/) (API compatível com OpenAI em `/v1`), orquestrado com LangChain.

## Como funciona

```
pergunta ──► planner ──► router ──► answer (por sub-pergunta) ──► synthesize ──► relatório
              │            │             │
              │            │             ├─ precisa_busca=true : busca → filtro de relevância → LLM com evidências
              │            │             └─ precisa_busca=false: LLM local, marcado como [sem fonte externa]
              │            └─ decide quais sub-perguntas exigem busca externa
              └─ gera até `max_subquestions` (5) sub-perguntas em JSON
```

1. **Planner** – decompõe a pergunta em sub-perguntas autocontidas (o limite é um teto, não uma meta; perguntas simples geram uma só).
2. **Router** – classifica cada sub-pergunta como "precisa de busca" ou não (ex.: perguntas sobre o próprio assistente, cálculo puro, opinião). Na dúvida, busca. Se o router falhar, todas as sub-perguntas são pesquisadas.
3. **Answer** – para as pesquisáveis: consulta o backend de busca, descarta trechos irrelevantes (filtro léxico, ver abaixo) e responde citando o título da fonte entre colchetes. Se não há fonte relevante, devolve `[nao foi possivel apurar: ...]` em vez de inventar.
4. **Synthesize** – escreve a resposta final usando somente o que foi apurado; o que não foi verificado é declarado explicitamente, e respostas `[sem fonte externa]` não são apresentadas como apuradas.

As saídas do planner e do router são JSON (`json_object`) validadas por Pydantic: o LLM sugere, o schema garante.

### Filtro de relevância

`src/utils/relevance.py` calcula, para cada trecho recuperado, a fração dos termos de conteúdo da sub-pergunta (sem acento, stopwords PT/EN e tokens curtos removidos) presentes no título + texto. Trechos abaixo de `RELEVANCE_THRESHOLD` (padrão `0.4`) são descartados.

## Estrutura

```
config/.env.test            # exemplo de configuração (copie para config/.env)
src/
├── main.py                 # loop interativo (CLI)
├── models/agent.py         # DeepResearchAgent: plan / route / answer / synthesize / run
├── schemas/schemas.py      # modelos Pydantic (sub-perguntas, decisões do router)
└── utils/
    ├── search.py           # backends de busca plugáveis
    ├── relevance.py        # filtro de relevância léxico
    └── templates.py        # prompts (planner, router, local, answer, synthesis)
```

## Backends de busca

Escolhidos por `SEARCH_BACKEND`:

| Backend      | Requisitos                                        | Observações |
|--------------|---------------------------------------------------|-------------|
| `wikipedia`  | nenhum (usa `requests`)                           | Wikipedia em português, 3 resultados, introdução de cada página |
| `duckduckgo` | `pip install -U ddgs langchain-community`         | Padrão se `SEARCH_BACKEND` não for definido; sujeito a rate limit |
| `tavily`     | `pip install tavily-python` e `TAVILY_API_KEY`    | Pensado para RAG; exige Python 3.9+ |

Para adicionar um backend, crie uma classe com `name`, `search(query)` (retornando `[{title, url, text}]`) e registre-a em `BACKENDS` em `src/utils/search.py`.

## Instalação

Não há `requirements.txt`; as dependências usadas pelo código são:

```bash
pip install langchain-openai langchain-core pydantic python-dotenv requests
# opcionais, conforme o backend escolhido:
pip install -U ddgs langchain-community   # duckduckgo
pip install tavily-python                 # tavily
```

Requer um servidor Ollama em execução com o modelo desejado já baixado (`ollama pull <modelo>`).

## Configuração

Copie o exemplo e ajuste:

```bash
cp config/.env.test config/.env
```

| Variável               | Padrão                     | Descrição |
|------------------------|----------------------------|-----------|
| `OLLAMA_MODEL_NAME`    | –                          | Nome do modelo no Ollama |
| `OLLAMA_HOST`          | `http://localhost:11434/`  | URL do servidor Ollama |
| `SEARCH_BACKEND`       | `duckduckgo`               | `wikipedia`, `duckduckgo` ou `tavily` |
| `RELEVANCE_THRESHOLD`  | `0.4`                      | Score mínimo (0–1) para uma fonte ser usada |
| `enable_thinking`      | `false`                    | Se `false`, adiciona ` /no_think` aos prompts (Qwen3) para o bloco `<think>` não poluir o JSON |
| `OLLAMA_API_KEY`       | `ollama`                   | Ignorada pelo Ollama, exigida pelo cliente |
| `TAVILY_API_KEY`       | –                          | Necessária apenas para o backend `tavily` |

`config/.env` está no `.gitignore`.

## Uso

Execute a partir da raiz do projeto (o `.env` é lido de `./config/.env`):

```bash
python src/main.py
```

Digite a pergunta no prompt; `exit`/`quit` (ou Ctrl+C / Ctrl+D) encerra. Os logs mostram o plano, as decisões do router e quantas fontes passaram no filtro de relevância por sub-pergunta.

## Observações

- Os prompts e as mensagens do sistema estão em português.
- O `DeepResearchAgent` não guarda estado e expõe uma operação por método, o que facilita encaixá-lo como nós de um grafo (ex.: LangGraph) no futuro.
- Testes devem usar o backend `wikipedia` (regra do projeto, ver `CLAUDE.md`). Ainda não há suíte de testes no repositório.
