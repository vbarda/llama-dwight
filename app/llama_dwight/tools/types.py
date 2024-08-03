import enum

from langchain_core.pydantic_v1 import BaseModel, Field


class AggregationFunc(str, enum.Enum):
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    MEAN = "mean"
    MEDIAN = "median"


class AggregationInput(BaseModel):
    columns: list[str] = Field(description="List of columns to perform aggregation on")
    aggregation_func: AggregationFunc = Field(
        description="Aggregation function to apply to the columns"
    )
