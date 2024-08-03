from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langgraph.graph.state import CompiledStateGraph

from llama_dwight.tools.base import BaseDataToolKit


def make_qa_agent(llm: BaseChatModel, toolkit: BaseDataToolKit) -> CompiledStateGraph:
    schema = toolkit.get_schema()
    system_prompt = f"You are an experienced data analyst that has access to a dataset with the following schema: {schema}"
    return create_react_agent(llm, toolkit.get_tools(), state_modifier=system_prompt)
