import os

# somewhat hacky way to determine if the app is being run from inside LangGraph API
IS_LANGGRAPH_API = os.environ.get("POSTGRES_URI")
