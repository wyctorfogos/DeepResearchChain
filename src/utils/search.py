"""
    Backends de busca plugaveis.
"""
from __future__ import annotations

import os
import requests


class SearchError(Exception):
    pass


def format_results(results):
    """Formata resultados estruturados como texto para o contexto do LLM."""
    return "\n\n".join(
        f"[{r['title']}] ({r['url']})\n{r['text']}" for r in results
    )


class WikipediaSearch:
    """
        MediaWiki API. Sem chave e sem dependencia extra (usa requests).
    """

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

    def search(self, query):
        try:
            found = self._get({
                "action": "query", "list": "search",
                "srsearch": query, "srlimit": self.max_results,
            })["query"]["search"]
        except Exception as e:
            raise SearchError(f"busca wikipedia falhou: {e}")

        if not found:
            return []

        pageids = [str(p["pageid"]) for p in found]
        try:
            pages = self._get({
                "action": "query", "prop": "extracts",
                "explaintext": 1, "exintro": 1,
                "pageids": "|".join(pageids),
            })["query"]["pages"]
        except Exception as e:
            raise SearchError(f"leitura wikipedia falhou: {e}")

        results = []
        for pid in pageids:
            page = pages.get(pid, {})
            extract = (page.get("extract") or "").strip()
            if not extract:
                continue
            results.append({
                "title": page.get("title", "?"),
                "url": f"https://{self.lang}.wikipedia.org/?curid={pid}",
                "text": extract[:self.chars_per_page],
            })
        return results

    def run(self, query):
        return format_results(self.search(query))


class DuckDuckGoSearch:
    """
        Busca web aberta via ddgs. Sem chave. Sujeita a rate limit (202) sem proxy.
    """

    name = "duckduckgo"

    def __init__(self, max_results=5, region="wt-wt"):
        try:
            from langchain_community.tools import DuckDuckGoSearchResults
            from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
        except ImportError:
            raise SearchError(
                "pacote ausente: pip install -U ddgs langchain-community"
            )
        wrapper = DuckDuckGoSearchAPIWrapper(region=region, max_results=max_results)
        self._tool = DuckDuckGoSearchResults(
            api_wrapper=wrapper, output_format="list"
        )

    def search(self, query):
        try:
            results = self._tool.invoke(query)
        except Exception as e:
            raise SearchError(f"busca duckduckgo falhou: {e}")
        if not isinstance(results, list):
            raise SearchError(
                f"formato inesperado do duckduckgo: {type(results).__name__}"
            )
        return [
            {"title": r.get("title", "?"),
             "url": r.get("link", ""),
             "text": r.get("snippet", "")}
            for r in results if r.get("snippet")
        ]

    def run(self, query):
        return format_results(self.search(query))


class TavilySearch:
    """Busca feita para RAG - devolve texto limpo. Requer TAVILY_API_KEY.

    Melhor opcao para deep research: fontes separadas, texto substancial,
    agnostica de idioma. Exige Python 3.9+ (o pacote usa dict[...] em runtime).
    """

    name = "tavily"

    def __init__(self, max_results=5):
        try:
            from tavily import TavilyClient
        except ImportError:
            raise SearchError("pacote ausente: pip install tavily-python")
        key = os.getenv("TAVILY_API_KEY")
        if not key:
            raise SearchError("TAVILY_API_KEY nao definida")
        self._client = TavilyClient(api_key=key)
        self.max_results = max_results

    def search(self, query):
        try:
            res = self._client.search(query, max_results=self.max_results)
        except Exception as e:
            raise SearchError(f"busca tavily falhou: {e}")
        return [
            {"title": r.get("title", "?"), "url": r.get("url", ""),
             "text": r.get("content", "")}
            for r in res.get("results", []) if r.get("content")
        ]

    def run(self, query):
        return format_results(self.search(query))


BACKENDS = {
    "wikipedia": WikipediaSearch,
    "duckduckgo": DuckDuckGoSearch,
    "tavily": TavilySearch,
}


def get_search_backend(name=None):
    """
        Resolve o backend por nome, por SEARCH_BACKEND, ou usa duckduckgo.
    """
    name = (name or os.getenv("SEARCH_BACKEND") or "duckduckgo").lower()
    if name not in BACKENDS:
        raise SearchError(
            f"backend '{name}' desconhecido. Opcoes: {', '.join(BACKENDS)}"
        )
    return BACKENDS[name]()