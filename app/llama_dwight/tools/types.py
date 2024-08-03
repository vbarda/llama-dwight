import enum

from langchain_core.pydantic_v1 import BaseModel, Field


class AggregationFunc(str, enum.Enum):
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    MEAN = "mean"
    MEDIAN = "median"


def validate_aggregation_func(aggregation_func: AggregationFunc) -> None:
    if aggregation_func not in AggregationFunc:
        allowed_values = [value.value for value in AggregationFunc]
        raise ValueError(
            f"Expected aggregation_func to be one of '{allowed_values}', got '{aggregation_func}' instead."
        )


class AggregationInput(BaseModel):
    columns: list[str] = Field(description="List of columns to perform aggregation on")
    aggregation_func: AggregationFunc = Field(
        description="Aggregation function to apply to the columns. REMEMBER: Average always refers to 'mean'"
    )


class GroupbyInput(BaseModel):
    groupby_columns: list[str] = Field(description="List of columns to group by")
    value_column: str = Field(
        description="Column whose values will be aggregated for each group"
    )
    aggregation_func: AggregationFunc = Field(
        description="Aggregation function to apply to the value column, for each group. REMEMBER: Average always refers to 'mean'"
    )
