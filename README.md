# llama-dwight

## Assistant (to the) Regional Data Science Manager

![llama_dwight_small](https://github.com/user-attachments/assets/f12ac075-1d0c-492a-96e0-26d393f6fad0)

Llama Dwight an app for performing exploratory data analysis with Pandas & SQL. You can use natural language queries to ask questions about your dataset, and approve agent-suggested data analysis plan to ensure you get the most precise answer.

Llama Dwight is built using:
- Groq Llama3.1 models (via [LangChain ChatGroq](https://python.langchain.com/v0.2/docs/integrations/chat/groq/))
- [LangGraph](https://github.com/langchain-ai/langgraph) for building agents
- [LangGraph Cloud](https://langchain-ai.github.io/langgraph/cloud/) / [LangGraph Studio](https://github.com/langchain-ai/langgraph-studio) for serving and interacting with the agents locally

The app consists of 2 agents (Pandas and SQL)

Each agent has two main components:

- `llm` -- chat model that orchestrates the agent and interacts with the tools
- `Toolkit` – universal interface that supports the following tools
  - `.filter`
  - `.sort`
  - `.aggregate`
  - `.groupby_aggregate`

Each agent consists of two main steps:

- `create_plan` -- create a plan for data analysis that can be reviewed and modified by human
- `qa_agent` -- question-answering agent (think junior data analyst). This agent is implementedd as a ReAct-style agent that has access to the tools from the `Toolkit`.

Currently supported functionality:

- Filtering the data based on value (string, date, numeric values)
- Aggregating the data (sum, count, min, max, mean)
- Groupby aggregations (sum, count, min, max, mean)
- Sorting + top largest / smallest values

Example queries:

- What are 3 states with the largest avg sales in the west?
- Which product category is the most popular?
- What is the average order value for each of the customer segments?
- What was maximum monthly total sales amount in 2017
- What were the average quarterly sales in New York in 2018?

## Usage

First, navigate to `/app` and run `poetry install --with dev`

### Pandas Agent

To interact with the agent you can use the code below. You can also use 

```python
from dotenv import load_dotenv
from langgraph.checkpoint import MemorySaver

from llama_dwight.tools.pandas import PandasDataToolKit
from llama_dwight.llms import get_llm, LLMName
from llama_dwight.agents.pandas_analyst_agent import PandasAnalystAgent

_ = load_dotenv()

df = pd.read_csv("data.csv")

llm_groq_70b = get_llm(LLMName.GROQ_LLAMA_3_1_70B)

checkpointer = MemorySaver()
toolkit = PandasDataToolKit(df)
pandas_agent = PandasAnalystAgent(llm_groq_70b, toolkit, checkpointer)
pandas_agent_graph = pandas_agent.compile()

input_ = {"messages": [("human", "What are 3 states with the largest avg sales in the west?")]}
config = {"configurable": {"thread_id": "1"}}
result = pandas_agent_graph.invoke(input_, config)

# review the message
result["messages"][-1].pretty_print()

# update
updated_message = result["messages"][-1]
updated_message.content = "UPDATE THE PLAN HERE"
pandas_agent_graph.update_state(config, {"messages": [updated_message]})

# continue execution from here
final_result = pandas_agent_graph.invoke(None, config)
```

## Usage (server)

If you want to interact with the app, you would need to have an active LangSmith account, as the app requires LangGraph Cloud.
Follow instructions [here]https://docs.smith.langchain.com/how_to_guides/setup/create_account_api_key.

Then download [LangGraph Studio](https://github.com/langchain-ai/langgraph-studio). In the Studio, select the project (`/app` directory from this repo) and choose `pandas_agent`. Then you can interact with the graph by specifying `messages` input with the user query.
```
