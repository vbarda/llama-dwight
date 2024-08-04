from typing import Any, Coroutine, Union, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables.config import (
    RunnableConfig,
    get_config_list,
    get_executor_for_config,
)
from langchain_core.messages import AIMessage, AnyMessage, ToolCall, ToolMessage
from langgraph.prebuilt.chat_agent_executor import (
    AgentState,
    _get_model_preprocessing_runnable,
)
from langgraph.graph.state import CompiledStateGraph, StateGraph, END
from langgraph.pregel.types import RetryPolicy
from langgraph.prebuilt.tool_node import ToolNode

from llama_dwight.tools.base import BaseDataToolKit
from llama_dwight.tools.types import ToolName

SYSTEM_PROMPT = """You are an experienced data analyst that has access to a dataset with the following schema: {schema}."
You are given a question about the dataset and a detailed step-by-step plan to answer it.

REMEMBER:
- Only use column names from the originally provided schema.
- Only respond with the answer to the original question, do not mention the tools you used.

Now start with the first tool you need to call. Begin!"""

INCORRECT_TOOL_ORDER_ERROR_MESSAGE = (
    "Error: incorrect tool calls -- cannot have both `aggregate` and `groupby`"
)
INVALID_TOOL_NAME_ERROR_TEMPLATE = (
    "Error: {requested_tools} is not a valid tool, try one of [{available_tools}]."
)


def preprocess_tool_calls(tool_calls: list[ToolCall]) -> tuple[list[ToolCall], bool]:
    """Enforce expected order on tool calls, if received multiple."""
    filter_tool_calls = []
    sort_tool_calls = []
    aggregate_tool_calls = []
    groupby_tool_calls = []
    is_valid = True
    for tool_call in tool_calls:
        if tool_call["name"] == ToolName.FILTER:
            filter_tool_calls.append(tool_call)
        elif tool_call["name"] == ToolName.SORT:
            sort_tool_calls.append(tool_call)
        elif tool_call["name"] == ToolName.AGGREGATE:
            aggregate_tool_calls.append(tool_call)
        elif tool_call["name"] == ToolName.GROUPBY:
            groupby_tool_calls.append(tool_call)
        else:
            pass

    if aggregate_tool_calls and groupby_tool_calls:
        is_valid = False

    preprocessed_tool_calls = [
        *filter_tool_calls,
        *sort_tool_calls,
        *aggregate_tool_calls,
        *groupby_tool_calls,
    ]
    return preprocessed_tool_calls, is_valid


class SequentialToolNode(ToolNode):
    """A version of ToolNode that runs multiple tools in a pre-defined, sequential order for data processing."""

    def _func(
        self, input: Union[list[AnyMessage], dict[str, Any]], config: RunnableConfig
    ) -> Any:
        tool_calls, output_type = self._parse_input(input)
        if output_type == "list":
            raise ValueError("Output type 'list' is not currently supported.")

        requested_tools = set(call["name"] for call in tool_calls)
        available_tools = {value.value for value in ToolName}
        if unknown_tools := requested_tools - available_tools:
            INVALID_TOOL_NAME_ERROR_TEMPLATE.format(
                requested_tools=unknown_tools, available_tools=available_tools
            )
            return {
                "messages": [
                    ToolMessage(
                        INCORRECT_TOOL_ORDER_ERROR_MESSAGE,
                        name=call["name"],
                        tool_call_id=call["id"],
                    )
                    for call in tool_calls
                    if call["name"] in unknown_tools
                ]
            }

        if len(requested_tools) == 1:
            config_list = get_config_list(config, len(tool_calls))
            with get_executor_for_config(config) as executor:
                outputs = [*executor.map(self._run_one, tool_calls, config_list)]
        else:
            preprocessed_tool_calls, are_tool_calls_valid = preprocess_tool_calls(
                tool_calls
            )
            if not are_tool_calls_valid:
                return {
                    "messages": [
                        ToolMessage(
                            INCORRECT_TOOL_ORDER_ERROR_MESSAGE,
                            name=call["name"],
                            tool_call_id=call["id"],
                        )
                        for call in preprocessed_tool_calls
                    ]
                }

            config_list = get_config_list(config, len(preprocessed_tool_calls))
            tool_output = None
            for tool_call, tool_config in zip(preprocessed_tool_calls, config_list):
                tool_output = self._run_one(tool_call, tool_config)
                # hacky way to exit early on error
                if "error" in tool_output.content:
                    break

            outputs = [tool_output]
        return {"messages": outputs}

    def _afunc(
        self, input: Union[list[AnyMessage], dict[str, Any]], config: RunnableConfig
    ) -> Coroutine[Any, Any, Any]:
        raise NotImplementedError


def make_qa_agent(llm: BaseChatModel, toolkit: BaseDataToolKit) -> CompiledStateGraph:
    """Make question-answering agent that uses an external data toolkit (pandas or DB)."""
    schema = toolkit.get_schema()
    tools = toolkit.get_tools()

    # add system message
    preprocessor = _get_model_preprocessing_runnable(
        SYSTEM_PROMPT.format(schema=schema), None
    )
    model_runnable = preprocessor | llm.bind_tools(tools)

    def call_model(
        state: AgentState,
        config: RunnableConfig,
    ) -> AgentState:
        response = model_runnable.invoke(state, config)
        if state["is_last_step"] and response.tool_calls:
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, need more steps to process this request.",
                    )
                ]
            }
        return {"messages": [response]}

    def should_continue(state: AgentState) -> Literal["continue", "end"]:
        messages = state["messages"]
        last_message = messages[-1]
        if not last_message.tool_calls:
            return "end"
        else:
            return "continue"

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model, retry=RetryPolicy(max_attempts=2))
    # this is the only thing that's different from create_react_agent
    workflow.add_node("tools", SequentialToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")
    return workflow.compile()
