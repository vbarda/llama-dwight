import enum
import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama


@enum.unique
class LLMName(str, enum.Enum):
    OLLAMA_3_1_8B = "llama3.1"
    GROQ_LLAMA_3_1_8B = "llama-3.1-8b-instant"
    GROQ_LLAMA_3_1_70B = "llama-3.1-70b-versatile"


OLLAMA_MODELS = frozenset({LLMName.OLLAMA_3_1_8B})
GROQ_MODELS = frozenset({LLMName.GROQ_LLAMA_3_1_8B, LLMName.GROQ_LLAMA_3_1_70B})


def get_llm(name: LLMName, local: bool = False) -> BaseChatModel:
    if name in OLLAMA_MODELS:
        host_ip = "localhost" if local else os.environ.get("HOST_IP", "not set")
        ollama_url = f"http://{host_ip}:11434"
        return ChatOllama(model=name, temperature=0, base_url=ollama_url)
    elif name in GROQ_MODELS:
        return ChatGroq(model=name, temperature=0)
    else:
        raise ValueError(f"Unsupported model '{name}'")
