from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langgraph.graph.state import CompiledStateGraph

from llama_dwight.tools.base import BaseDataToolKit

SYSTEM_PROMPT = """You are an experienced data analyst that has access to a dataset with the following schema: {schema}."
When answering complex questions, think step by step. Break the problem down into a series of the following steps:
- first, apply filters (if any)
- then, apply sort + limit (if any)
- finally, apply aggregations

Always use a single tool at a time.
Only use column names from the originally provided schema.

Now start with the first tool you need to call. Begin!"""


def make_qa_agent(llm: BaseChatModel, toolkit: BaseDataToolKit) -> CompiledStateGraph:
    schema = toolkit.get_schema()
    return create_react_agent(
        llm, toolkit.get_tools(), state_modifier=SYSTEM_PROMPT.format(schema=schema)
    )
