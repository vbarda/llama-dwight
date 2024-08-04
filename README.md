# llama-dwight

## Assistant (to the) Regional Data Science Manager

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