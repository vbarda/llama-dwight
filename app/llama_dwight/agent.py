from langchain_core.tools import tool

from langgraph.prebuilt import create_react_agent

from llama_dwight.llms import LLMName, get_llm

llm = get_llm(LLMName.GROQ_LLAMA_3_1_8B)


@tool
def get_weather(city: str) -> str:
    """Use this to look up weather."""
    if city.lower() == "nyc" or city.lower() == "new york":
        return "Amazing, sunny and 80 degrees"
    return "Not great, cloudy and 60 degrees"


tools = [get_weather]
graph = create_react_agent(llm, tools)
