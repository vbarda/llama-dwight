from langgraph.graph.message import MessagesState

from llama_dwight.agents.analyst_agent import AnalystAgent
from llama_dwight.agents.qa_agent import make_qa_agent
from llama_dwight.llms import LLMName, get_llm
from llama_dwight.tools.pandas import PandasDataToolKit

DEFAULT_FILEPATH = "data.csv"
llm = get_llm(LLMName.GROQ_LLAMA_3_1_70B)


class PandasAnalystState(MessagesState):
    # this serves as an interface for a user to specify the filepath to a CSV
    # that will be loaded as a dataframe
    filepath: str


class PandasAnalystAgent(AnalystAgent):
    state_schema = PandasAnalystState

    def load_data_toolkit(self, state: PandasAnalystState) -> PandasAnalystState:
        if self.data_toolkit is not None:
            # clear toolkit state and continue
            self.data_toolkit.clear()
            return state

        filepath = state["filepath"] or DEFAULT_FILEPATH
        data_toolkit = PandasDataToolKit.from_filepath(filepath, preprocess=True)
        self.data_toolkit = data_toolkit
        self.qa_agent = make_qa_agent(self.llm, data_toolkit)
        return state


graph = PandasAnalystAgent(llm).compile()
