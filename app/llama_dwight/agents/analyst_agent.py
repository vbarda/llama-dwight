from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph.state import StateGraph, END, CompiledStateGraph
from langgraph.checkpoint import BaseCheckpointSaver
from langgraph.graph.message import MessagesState

from llama_dwight.tools.base import BaseDataToolKit
from llama_dwight.agents.qa_agent import make_qa_agent


PLAN_SYSTEM_PROMPT = """You are an experienced data analyst that has access to a dataset with the following schema: {schema}."
You need to help a junior data analyst answer the following question: {question}.
Junior analyst has access to the following tools: {available_tools}."""

PLAN_MESSAGE = """Write a step-by-step plan for the junior analyst to answer the question.
If the plan has multiple steps, the steps should be in the following order:
- first, use the `filter` tool (if relevant)
- then, use the `sort` tool (if any)
- finally, use `aggregate` or `groupby` tool. Never use `aggregate` tool after `groupby`."""


class AnalystAgent:
    state_schema: MessagesState

    def __init__(
        self,
        llm: BaseChatModel,
        data_toolkit: Optional[BaseDataToolKit] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ) -> None:
        self.llm = llm
        self.data_toolkit = data_toolkit
        self.qa_agent = (
            None if data_toolkit is None else make_qa_agent(llm, data_toolkit)
        )
        self.checkpointer = checkpointer

    def load_data_toolkit(self, state: MessagesState) -> MessagesState:
        raise NotImplementedError

    def create_plan(self, state: MessagesState) -> MessagesState:
        if self.data_toolkit is None:
            raise ValueError("Cannot create a plan without a data toolkit")

        schema = self.data_toolkit.get_schema()
        question = state["messages"][-1].content
        available_tools = [tool.name for tool in self.data_toolkit.get_tools()]
        system_message = SystemMessage(
            content=PLAN_SYSTEM_PROMPT.format(
                schema=schema, question=question, available_tools=available_tools
            )
        )
        human_message = HumanMessage(content=PLAN_MESSAGE)
        response = self.llm.invoke([system_message, human_message])
        return {"messages": [response]}

    def call_qa_agent(self, state: MessagesState) -> MessagesState:
        if self.qa_agent is None:
            raise ValueError("Question-answering agent hasn't been initialized.")

        qa_agent_response = self.qa_agent.invoke(state)
        return {"messages": qa_agent_response["messages"]}

    def compile(self) -> CompiledStateGraph:
        workflow = StateGraph(self.state_schema)
        workflow.add_node("load_data", self.load_data_toolkit)
        workflow.add_node("create_plan", self.create_plan)
        workflow.add_node("qa_agent", self.call_qa_agent)
        workflow.set_entry_point("load_data")
        workflow.add_edge("load_data", "create_plan")
        workflow.add_edge("create_plan", "qa_agent")
        workflow.add_edge("qa_agent", END)
        return workflow.compile(
            interrupt_before=["qa_agent"], checkpointer=self.checkpointer
        )
