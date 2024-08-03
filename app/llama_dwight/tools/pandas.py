from typing import Any, Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

import pandas as pd

from llama_dwight import config
from llama_dwight.tools.types import AggregationFunc
from llama_dwight.shared import IN_MEMORY_STORE, DATAFRAME_KEY


def get_dataframe(state: dict) -> pd.DataFrame:
    # we can't pass dataframes around in the state as dataframes are not serializable
    # and custom serializer is not an option for using it with LangGraph API / Studio
    # so we are using an in-memory dataframe instead
    if config.IS_LANGGRAPH_API:
        df: pd.DataFrame = IN_MEMORY_STORE.get(DATAFRAME_KEY)
    # for interactive workflows in the notebook we are using dataframe from the state
    else:
        df = state["df"]

    if df is None:
        raise ValueError("Dataframe is not loaded.")
    return df


def get_schema(state: dict) -> dict:
    df = get_dataframe(state)
    return df.dtypes.astype("str").to_dict()


@tool
def aggregate(
    columns: list[str],
    aggregation_func: AggregationFunc,
    state: Annotated[dict, InjectedState],
) -> dict[str, Any]:
    """Aggregate column values.

    Args:
        columns: List of columns to perform aggregation on
        aggregation_func: Aggregation function to apply to the columns. REMEMBER: Average always refers to 'mean'
    """
    df = get_dataframe(state)
    if not isinstance(columns, list):
        raise TypeError(f"Expected columns to be a list, got '{columns}' instead")

    if aggregation_func not in AggregationFunc:
        allowed_values = [value.value for value in AggregationFunc]
        raise ValueError(
            f"Expected aggregation_func to be one of '{allowed_values}', got '{aggregation_func}' instead."
        )

    # TODO: figure out if we need to support a different spec of [(column, aggregation_func),...] pairs
    # easy to support in pandas / SQL
    return df[columns].agg(aggregation_func).to_dict()
