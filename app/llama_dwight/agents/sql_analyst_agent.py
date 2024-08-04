from langgraph.graph.message import MessagesState

from llama_dwight.agents.analyst_agent import AnalystAgent
from llama_dwight.agents.qa_agent import make_qa_agent
from llama_dwight.llms import LLMName, get_llm
from llama_dwight.tools.sql import SQLDataToolKit

DEFAULT_CONN_STRING = "sqlite:///data.db"
llm = get_llm(LLMName.GROQ_LLAMA_3_1_70B)


class SQLAnalystState(MessagesState):
    # this serves as an interface for a user to specify the conn str for the DB
    db_conn_string: str
    table: str


class SQLAnalystAgent(AnalystAgent):
    state_schema = SQLAnalystState

    def load_data_toolkit(self, state: SQLAnalystState) -> SQLAnalystState:
        if self.data_toolkit is not None:
            # clear toolkit state and continue
            self.data_toolkit.clear()
            return state

        conn_string = state["db_conn_string"] or DEFAULT_CONN_STRING
        data_toolkit = SQLDataToolKit.from_conn_string(conn_string, state["table"])
        self.data_toolkit = data_toolkit
        self.qa_agent = make_qa_agent(self.llm, data_toolkit)
        return state


graph = SQLAnalystAgent(llm).compile()
