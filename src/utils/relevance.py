
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
MIN_STEM_LEN = 4

# Sufixos (sem acento) do mais longo ao mais curto. Radical minimo de 4 letras
# evita estragar palavras curtas. Stemmer leve e propositalmente conservador:
# o objetivo e casar flexoes ("escritos"/"escrito", "fundacao"/"fundacoes"),
# nao fazer analise morfologica.
SUFFIXES = sorted([
    # portugues
    "amentos", "imentos", "amento", "imento", "adores", "acoes", "icoes",
    "idades", "mente", "ismos", "istas", "ador", "coes", "cao", "oes", "ismo",
    "ista", "idade", "ados", "adas", "idos", "idas", "ado", "ada", "ido",
    "ida", "ais", "eis", "es", "as", "os",
    # ingles
    "ations", "ation", "ings", "ing", "ies", "ed", "ly",
    # plural generico
    "s",
], key=len, reverse=True)


def stem(token):
    """Remove o primeiro sufixo conhecido e a vogal final (a/e/o), sempre
    mantendo um radical >= MIN_STEM_LEN."""
    for suf in SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= MIN_STEM_LEN:
            token = token[: -len(suf)]
            break
    if token[-1:] in ("a", "e", "o") and len(token) - 1 >= MIN_STEM_LEN:
        token = token[:-1]
    return token


def normalize(text):
    """Minusculas, sem acento - 'Vitória' e 'vitoria' devem casar."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower()


def content_terms(text):
    """Termos de conteudo: sem acento, sem stopword, sem token curto, em radical."""
    tokens = re.findall(r"[a-z0-9]+", normalize(text))
    return {
        stem(t) for t in tokens
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
