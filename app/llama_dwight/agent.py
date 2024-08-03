from langchain_groq import ChatGroq
from langchain_core.tools import tool

from langgraph.prebuilt import create_react_agent

llm = ChatGroq(model="llama-3.1-8b-instant")


@tool
def get_weather(city: str) -> str:
    """Use this to look up weather."""
    if city.lower() == "nyc" or city.lower() == "new york":
        return "Amazing, sunny and 80 degrees"
    return "Not great, cloudy and 60 degrees"


tools = [get_weather]
graph = create_react_agent(llm, tools)
