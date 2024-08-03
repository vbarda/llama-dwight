from langgraph.prebuilt.chat_agent_executor import AgentState, create_react_agent

import pandas as pd

from llama_dwight import config
from llama_dwight.llms import LLMName, get_llm
from llama_dwight.tools.pandas import aggregate

llm = get_llm(LLMName.OLLAMA_3_1_8B)


if config.IS_LANGGRAPH_API:
    # we can't pass dataframes around in the state as dataframes are not serializable
    # and custom serializer is not an option for using it with LangGraph API / Studio
    # so we are using an in-memory dataframe instead. this state is just the default AgentState
    class CustomAgentState(AgentState):
        pass
else:
    # for interactive workflows in the notebook we are using dataframe from the state
    class CustomAgentState(AgentState):
        df: pd.DataFrame


tools = [aggregate]
graph = create_react_agent(llm, tools, state_schema=CustomAgentState)
