from typing import Dict, List, Any
from pydantic import BaseModel, Field

class InputState(BaseModel):
    input_query:str

class SubPerguntas(BaseModel):
    """Uma sub-pergunta de pesquisa, autocontida e pesquisavel na web."""
    id:int = Field(description="Indice sequencial comecando em 1")
    pergunta:str = Field(description="A sub-pergunta em linguagem natural")

class ListOfQuestions(BaseModel):
    """Decomposicao de uma pergunta ampla em sub-perguntas de pesquisa."""
    ListOfQueries:List[SubPerguntas] = Field(description="As sub-perguntas geradas")
