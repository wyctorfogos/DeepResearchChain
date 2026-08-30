"""
    Filtro de relevancia lexico entre a sub-pergunta e os trechos recuperados.
"""
import re
import unicodedata

STOPWORDS = {
    # portugues
    "a", "ao", "aos", "as", "com", "como", "da", "das", "de", "do", "dos", "e",
    "em", "entre", "era", "essa", "esse", "esta", "este", "eu", "foi", "for",
    "hoje", "isso", "ja", "la", "mais", "mas", "me", "mesmo", "meu", "na",
    "nao", "nas", "nem", "no", "nos", "num", "numa", "o", "os", "ou", "para",
    "pela", "pelo", "por", "qual", "quais", "quando", "que", "quem", "se",
    "sem", "ser", "seu", "sua", "sao", "tem", "um", "uma", "voce", "e",
    "ela", "ele", "elas", "eles", "aos", "ate", "ainda", "apenas", "cada",
    "sobre", "tambem", "todo", "toda", "todos", "todas", "muito", "onde",
    # ingles
    "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "was", "what", "when", "where", "which", "who", "why", "with", "your",
}

MIN_TERM_LEN = 3


def normalize(text):
    """Minusculas, sem acento - 'Vitória' e 'vitoria' devem casar."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower()


def content_terms(text):
    """Termos de conteudo: sem acento, sem stopword, sem token curto."""
    tokens = re.findall(r"[a-z0-9]+", normalize(text))
    return {
        t for t in tokens
        if len(t) >= MIN_TERM_LEN and t not in STOPWORDS
    }


def score(query, result):
    """Fracao dos termos de conteudo da pergunta presentes no trecho (0.0-1.0)."""
    q_terms = content_terms(query)
    if not q_terms:
        return 0.0
    d_terms = content_terms(f"{result.get('title','')} {result.get('text','')}")
    return len(q_terms & d_terms) / len(q_terms)


def filter_results(query, results, threshold=0.5):
    """Devolve [(result, score)] acima do limiar, do mais relevante ao menos."""
    scored = [(r, score(query, r)) for r in results]
    kept = [(r, s) for r, s in scored if s >= threshold]
    return sorted(kept, key=lambda rs: rs[1], reverse=True)
