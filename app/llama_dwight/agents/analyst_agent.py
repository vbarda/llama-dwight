from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import StateGraph, END, CompiledStateGraph
from langgraph.graph.message import MessagesState

from llama_dwight.tools.base import BaseDataToolKit
from llama_dwight.agents.qa_agent import make_qa_agent


class AnalystAgent:
    state_schema: MessagesState

    def __init__(
        self, llm: BaseChatModel, data_toolkit: Optional[BaseDataToolKit] = None
    ) -> None:
        self.llm = llm
        self.data_toolkit = data_toolkit
        self.qa_agent = (
            None if data_toolkit is None else make_qa_agent(llm, data_toolkit)
        )

    def load_data_toolkit(self, state: MessagesState) -> MessagesState:
        raise NotImplementedError

    def call_qa_agent(self, state: MessagesState) -> MessagesState:
        if self.qa_agent is None:
            raise ValueError("Question-answering agent hasn't been initialized.")

        qa_agent_response = self.qa_agent.invoke(state)
        return {"messages": qa_agent_response["messages"]}

    def compile(self) -> CompiledStateGraph:
        workflow = StateGraph(self.state_schema)
        workflow.add_node("load_data", self.load_data_toolkit)
        workflow.add_node("qa_agent", self.call_qa_agent)
        workflow.set_entry_point("load_data")
        workflow.add_edge("load_data", "qa_agent")
        workflow.add_edge("qa_agent", END)
        return workflow.compile()
