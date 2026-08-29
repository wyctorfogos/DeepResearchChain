"""Backends de busca plugaveis.

Cada backend expoe .name e .run(query) -> str. O texto retornado e o que entra
no contexto do LLM, entao cada trecho vem rotulado com titulo e URL para que a
etapa de sintese consiga citar a fonte.
"""
import os
import requests


class SearchError(Exception):
    pass


class WikipediaSearch:
    """MediaWiki API. Sem chave e sem dependencia extra (usa requests)."""

    name = "wikipedia"

    def __init__(self, lang="pt", max_results=3, chars_per_page=1200, timeout=15):
        self.endpoint = f"https://{lang}.wikipedia.org/w/api.php"
        self.max_results = max_results
        self.chars_per_page = chars_per_page
        self.timeout = timeout
        self.lang = lang

    def _get(self, params):
        params["format"] = "json"
        r = requests.get(self.endpoint, params=params, timeout=self.timeout,
                         headers={"User-Agent": "deep-research/0.1"})
        r.raise_for_status()
        return r.json()

    def run(self, query):
        try:
            found = self._get({
                "action": "query", "list": "search",
                "srsearch": query, "srlimit": self.max_results,
            })["query"]["search"]
        except Exception as e:
            raise SearchError(f"busca wikipedia falhou: {e}")

        if not found:
            return ""

        pageids = [str(p["pageid"]) for p in found]
        try:
            pages = self._get({
                "action": "query", "prop": "extracts",
                "explaintext": 1, "exintro": 1,
                "pageids": "|".join(pageids),
            })["query"]["pages"]
        except Exception as e:
            raise SearchError(f"leitura wikipedia falhou: {e}")

        trechos = []
        for pid in pageids:
            page = pages.get(pid, {})
            extract = (page.get("extract") or "").strip()
            if not extract:
                continue
            titulo = page.get("title", "?")
            url = f"https://{self.lang}.wikipedia.org/?curid={pid}"
            trechos.append(f"[{titulo}] ({url})\n{extract[:self.chars_per_page]}")
        return "\n\n".join(trechos)


class DuckDuckGoSearch:
    """Busca web aberta. Sujeita a rate limit (202) sem proxy."""

    name = "duckduckgo"

    def __init__(self):
        try:
            from langchain_community.tools import DuckDuckGoSearchRun
        except ImportError:
            raise SearchError("pacote ausente: pip install duckduckgo-search")
        self._tool = DuckDuckGoSearchRun()

    def run(self, query):
        try:
            return self._tool.run(query)
        except Exception as e:
            raise SearchError(f"busca duckduckgo falhou: {e}")


class TavilySearch:
    """Busca feita para RAG - devolve texto limpo. Requer TAVILY_API_KEY."""

    name = "tavily"

    def __init__(self, max_results=3):
        try:
            from tavily import TavilyClient
        except ImportError:
            raise SearchError("pacote ausente: pip install tavily-python")
        key = os.getenv("TAVILY_API_KEY")
        if not key:
            raise SearchError("TAVILY_API_KEY nao definida")
        self._client = TavilyClient(api_key=key)
        self.max_results = max_results

    def run(self, query):
        try:
            res = self._client.search(query, max_results=self.max_results)
        except Exception as e:
            raise SearchError(f"busca tavily falhou: {e}")
        return "\n\n".join(
            f"[{r.get('title','?')}] ({r.get('url','')})\n{r.get('content','')}"
            for r in res.get("results", [])
        )


BACKENDS = {
    "wikipedia": WikipediaSearch,
    "duckduckgo": DuckDuckGoSearch,
    "tavily": TavilySearch,
}


def get_search_backend(name=None):
    """Resolve o backend por nome, por SEARCH_BACKEND, ou usa wikipedia."""
    name = (name or os.getenv("SEARCH_BACKEND") or "wikipedia").lower()
    if name not in BACKENDS:
        raise SearchError(
            f"backend '{name}' desconhecido. Opcoes: {', '.join(BACKENDS)}"
        )
    return BACKENDS[name]()
