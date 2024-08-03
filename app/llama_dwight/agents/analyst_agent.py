from langchain_core.messages import SystemMessage
from langgraph.graph.state import StateGraph, END
from langgraph.graph.message import MessagesState

from llama_dwight.agents import qa_agent
from llama_dwight.shared import load_data
from llama_dwight.tools.pandas import get_schema

DEFAULT_FILEPATH = "data.csv"


# TODO: make a version of this agent that doesn't need in-memory store for usage outside of LangGraph studio
class AnalystState(MessagesState):
    # this serves as an interface for a user to specify the filepath
    filepath: str


class AnalystStateInput(MessagesState):
    filepath: str


def load_data_from_filepath(state: AnalystState) -> AnalystState:
    """Load data from filepath and set it globally."""
    filepath = state["filepath"] or DEFAULT_FILEPATH
    # this is going to modify the global in-memory store
    load_data(filepath)
    return state


def call_qa_agent(state: AnalystState) -> AnalystState:
    schema = get_schema(state)
    system_message = SystemMessage(
        content=f"You are an experienced data analyst that has access to a dataset with the following schema: {schema}"
    )
    messages = [system_message] + state["messages"]
    qa_agent_response = qa_agent.graph.invoke({"messages": messages})
    return {"messages": [qa_agent_response["messages"][-1]]}


workflow = StateGraph(AnalystState, input=AnalystStateInput)

workflow.add_node("load_data", load_data_from_filepath)
workflow.add_node("qa_agent", call_qa_agent)
workflow.set_entry_point("load_data")
workflow.add_edge("load_data", "qa_agent")
workflow.add_edge("qa_agent", END)

graph = workflow.compile()
