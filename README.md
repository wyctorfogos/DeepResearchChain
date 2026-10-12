# DeepResearchChain

Agente de *deep research* em linha de comando. Ele recebe uma pergunta ampla, a decompõe em sub-perguntas, pesquisa cada uma na web **em paralelo**, responde **apenas com base nas fontes encontradas**, avalia se o resultado basta (e refina a busca se não) e escreve um relatório com a lista de fontes.

O LLM roda localmente via [Ollama](https://ollama.com/) (API compatível com OpenAI em `/v1`) e o fluxo é orquestrado com [LangGraph](https://github.com/langchain-ai/langgraph).

## Como funciona

```text
START → plan → route ─┬─► research (sub-pergunta 1) ─┐
                ▲     ├─► research (sub-pergunta 2) ─┼─► evaluate ─┬─► write → END
                │     └─► research (sub-pergunta N) ─┘             │
                └───────── lacunas e rodadas restantes ◄───────────┘
```

1. **plan**: decompõe a pergunta em sub-perguntas autocontidas. Se o JSON vier inválido, tenta de novo (`PLAN_RETRIES`).
2. **route**: classifica cada sub-pergunta como "precisa de busca" ou não. Na dúvida, busca; se o router falhar, busca tudo.
3. **research**: uma tarefa por sub-pergunta, executadas em paralelo. As pesquisáveis passam por busca, filtro de relevância e um LLM que cita a fonte. As demais são respondidas localmente e marcadas `[sem fonte externa]`. Sem fonte relevante, o resultado é `[nao foi possivel apurar: ...]`.
4. **evaluate**: o LLM julga se o apurado basta. Se houver lacunas e restarem rodadas (`MAX_ROUNDS`), gera novas sub-perguntas, sem repetir as já feitas, e volta para **route**.
5. **write**: sintetiza o relatório em *streaming* e anexa a seção **Fontes**, montada em código só com as URLs realmente recuperadas.

As saídas de plan, route e evaluate são JSON validado por Pydantic: o LLM sugere, o schema garante.

### Filtro de relevância

`src/utils/relevance.py` calcula, para cada trecho recuperado, a fração dos termos de conteúdo da sub-pergunta (sem acento, sem stopwords PT/EN, sem tokens curtos e reduzidos a radical por um stemmer leve de sufixos PT/EN, para casar flexões como "escritos"/"escrito") presentes no título e no texto. Trechos abaixo de `RELEVANCE_THRESHOLD` (padrão `0.4`) são descartados.

## Estrutura

```text
config/.env.test            # exemplo de configuração (copie para config/.env)
pyproject.toml              # empacotamento e metadados (versão, dependências)
requirements.txt            # dependências para ambiente de desenvolvimento
src/
├── main.py                 # CLI (modo interativo ou pergunta única) com streaming
├── graph.py                # StateGraph do LangGraph
├── models/agent.py         # DeepResearchAgent: plan / route / answer / evaluate / synthesize
├── schemas/schemas.py      # modelos Pydantic
└── utils/
    ├── search.py           # backends de busca plugáveis + cache
    ├── relevance.py        # filtro de relevância léxico
    └── templates.py        # prompts
tests/                      # testes (Wikipedia real + LLM falso)
```

## Instalação

Requer **Python 3.10+** e um servidor Ollama em execução com o modelo já baixado (`ollama pull <modelo>`).

```bash
# Opção 1: desenvolvimento
pip install -r requirements.txt

# Opção 2: como pacote (cria o comando `deep-research`)
pip install -e ".[duckduckgo,tavily,dev]"
```

Os extras `duckduckgo`, `tavily` e `dev` são opcionais; instale só os que for usar.

## Configuração

```bash
cp config/.env.test config/.env
```

Cada variável tem uma flag equivalente na CLI, que tem prioridade sobre o `.env`.

| Variável              | Flag                    | Padrão                    | Descrição                                      |
|-----------------------|-------------------------|---------------------------|------------------------------------------------|
| `OLLAMA_MODEL_NAME`   | `--model`               | -                         | Modelo no Ollama                               |
| `OLLAMA_HOST`         | `--host`                | `http://localhost:11434/` | URL do servidor Ollama                         |
| `SEARCH_BACKEND`      | `--backend`             | `duckduckgo`              | `wikipedia`, `duckduckgo` ou `tavily`          |
| `MAX_ROUNDS`          | `--max-rounds`          | `2`                       | Máximo de rodadas (1 desliga a iteração)       |
| `MAX_SUBQUESTIONS`    | `--max-subquestions`    | `5`                       | Teto de sub-perguntas por rodada               |
| `TEMPERATURE`         | `--temperature`         | `0.1`                     | Temperatura do LLM                             |
| `RELEVANCE_THRESHOLD` | `--relevance-threshold` | `0.4`                     | Score mínimo (0 a 1) para uma fonte ser usada  |
| `PLAN_RETRIES`        | `--plan-retries`        | `2`                       | Tentativas do planner                          |
| `WIKIPEDIA_LANG`      | `--wikipedia-lang`      | `pt`                      | Idioma da Wikipedia                            |
| `SEARCH_CACHE_TTL`    | -                       | `3600`                    | Validade do cache de buscas, em segundos       |
| `enable_thinking`     | `--thinking`            | `false`                   | Mantém o bloco `<think>` (Qwen3)               |
| `OLLAMA_API_KEY`      | -                       | `ollama`                  | Ignorada pelo Ollama, exigida pelo cliente     |
| `TAVILY_API_KEY`      | -                       | -                         | Só para o backend `tavily`                     |

Com `enable_thinking=false`, os prompts recebem o sufixo `/no_think` para o bloco `<think>` não poluir o JSON.

## Uso

Execute a partir da raiz do projeto, porque o `.env` é lido de `./config/.env`.

```bash
# modo interativo ('exit' ou 'quit' encerra)
python src/main.py

# pergunta única
python src/main.py --backend wikipedia "Qual a diferença entre Machado de Assis e José de Alencar?"

# instalado como pacote
deep-research --max-rounds 3 --wikipedia-lang en "History of the printing press"
```

O relatório aparece token a token. Os logs mostram o plano, as decisões do router e quantas fontes passaram no filtro. A saída tem esta forma:

```text
======================================================================
<relatório em texto corrido, citando as fontes entre colchetes>

## Fontes
- [Machado de Assis](https://pt.wikipedia.org/?curid=...)
- [José de Alencar](https://pt.wikipedia.org/?curid=...)
======================================================================
```

## Backends de busca

| Backend      | Requisitos                                  | Observações                                                            |
|--------------|---------------------------------------------|------------------------------------------------------------------------|
| `wikipedia`  | nenhum (usa `requests`)                     | 3 resultados, introdução de cada página; faz retry em 429              |
| `duckduckgo` | `ddgs` e `langchain-community`              | Padrão se `SEARCH_BACKEND` não for definido; sujeito a rate limit      |
| `tavily`     | `tavily-python` e `TAVILY_API_KEY`          | Pensado para RAG                                                       |

Todos são embrulhados em um cache em memória com TTL. Para adicionar um backend, crie uma classe com `name` e `search(query)` (retornando `[{title, url, text}]`) e registre-a em `BACKENDS` em `src/utils/search.py`.

## CI

`.github/workflows/tests.yml` roda os testes a cada push na `main` e em pull requests (Python 3.10 e 3.12). Como dependem da Wikipedia, a execução tenta até 3 vezes para tolerar o erro 429.

## Testes

```bash
pip install pytest
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests
```

Os testes usam a Wikipedia real (regra do projeto) e um LLM falso, então não precisam do Ollama. A variável `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` evita que plugins globais incompatíveis quebrem o pytest. A Wikipedia pode devolver 429 em execuções seguidas; se um teste falhar por isso, rode de novo após alguns segundos.
