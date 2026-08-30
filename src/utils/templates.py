PLANNER_TEMPLATE = """Voce e um planejador de pesquisa. Decomponha a pergunta do usuario em ate \
{max_subquestions} sub-perguntas objetivas, autocontidas e pesquisaveis na web.
Cada sub-pergunta deve cobrir um aspecto diferente, sem repetir as outras.

{max_subquestions} e um TETO, nao uma meta. Uma pergunta simples e direta deve gerar \
UMA unica sub-pergunta. So decomponha em varias quando a pergunta realmente tiver \
aspectos independentes que exigem apuracao separada.

Responda SOMENTE com JSON valido, sem nenhum texto ao redor, no formato:
{{"ListOfQueries": [{{"id": 1, "pergunta": "..."}}]}}

Pergunta: {query}{think_flag}"""

ROUTER_TEMPLATE = """Voce classifica sub-perguntas de pesquisa. Para cada uma, decida se \
responde-la exige consultar fontes externas (web).

Marque "precisa_busca": false APENAS quando NENHUMA fonte externa poderia responder:
- perguntas sobre o proprio assistente (seu nome, o que voce e)
- calculo puro ou reformatacao de algo ja dado na pergunta
- pedidos de opiniao ou preferencia

Marque "precisa_busca": true para TODO o resto, incluindo fatos que voce acredita saber \
de cabeca - o objetivo do sistema e responder com fonte citavel, nao de memoria.
Na duvida, use true.

Responda SOMENTE com JSON valido, sem texto ao redor:
{{"decisoes": [{{"id": 1, "precisa_busca": true, "motivo": "..."}}]}}

Sub-perguntas:
{perguntas}{think_flag}"""

LOCAL_TEMPLATE = """Responda a pergunta abaixo diretamente, com seu proprio conhecimento.
Esta pergunta foi classificada como nao pesquisavel na web - nao existe fonte externa \
a consultar.

Seja conciso e direto. Se voce nao souber com seguranca, diga que nao sabe em vez de \
inventar.

Pergunta: {pergunta}{think_flag}"""

ANSWER_TEMPLATE = """Responda a pergunta usando SOMENTE os trechos abaixo.
Nao complete com conhecimento proprio. Se os trechos nao respondem a pergunta, \
diga exatamente: NAO ENCONTRADO NAS FONTES.
Cite o titulo da fonte entre colchetes ao afirmar cada fato.

Trechos:
{evidencia}

Pergunta: {pergunta}{think_flag}"""

SYNTHESIS_TEMPLATE = """Voce e um redator de pesquisa. Abaixo estao sub-perguntas e o que \
foi apurado para cada uma.

Escreva uma resposta unica e coerente para a pergunta original, usando SOMENTE o que \
foi apurado. Se algo nao foi apurado, diga explicitamente que nao foi possivel verificar \
em vez de completar com conhecimento proprio.

Itens marcados com [sem fonte externa] foram respondidos sem consulta a fontes, por \
serem perguntas que nenhuma fonte externa poderia responder. Use o conteudo deles \
normalmente, mas nao os apresente como apurados em fontes.

Responda apenas com base nos itens listados em "Apurado". Nao reintroduza aspectos da \
pergunta original que nao estejam ali.

Pergunta original: {query}

Apurado:
{findings}{think_flag}"""

# Prefixo das respostas geradas sem consulta a fontes. A sintese usa isso para
# distinguir o que tem lastro externo do que veio do conhecimento do modelo.
SEM_FONTE = "[sem fonte externa]"
