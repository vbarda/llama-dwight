import enum
from typing import Optional

from langchain_core.pydantic_v1 import BaseModel, Field


@enum.unique
class ToolName(str, enum.Enum):
    FILTER = "filter"
    SORT = "sort"
    AGGREGATE = "aggregate"
    GROUPBY = "groupby_aggregate"


@enum.unique
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


@enum.unique
class GroupbyFreq(str, enum.Enum):
    MONTHLY = "ME"
    QUARTERLY = "QE"
    YEARLY = "YE"


class GroupbyInput(BaseModel):
    groupby_columns: list[str] = Field(description="List of columns to group by")
    value_column: str = Field(
        description="Column whose values will be aggregated for each group"
    )
    aggregation_func: AggregationFunc = Field(
        description="Aggregation function to apply to the value column, for each group. REMEMBER: Average always refers to 'mean'"
    )
    freq: GroupbyFreq = Field(
        description="Frequency for grouping by a date column. Values: 'ME' = monthly, 'QE' = quarterly, 'YE' = yearly"
    )


class FilterOperator(str, enum.Enum):
    GREATER = ">"
    GREATER_OR_EQ = ">="
    LESS = "<"
    LESS_OR_EQ = "<="
    EQ = "="
    NEQ = "!="


class FilterValueType(str, enum.Enum):
    NUMBER = "number"
    STRING = "string"
    DATETIME = "datetime"


class FilterSpec(BaseModel):
    column: str = Field(description="Column to filter on")
    value: str = Field(description="Value to use as a filter")
    value_type: FilterValueType = Field(
        description="Value type to use for the filter value"
    )
    operator: FilterOperator = Field(description="Filter operator to use")


class FilterInput(BaseModel):
    filters: list[FilterSpec] = Field(
        description="List of filter specifications to filter data on"
    )


class SortInput(BaseModel):
    column: str = Field(description="Column to sort on")
    ascending: bool = Field(description="Whether to sort ascending or descending")
    limit: Optional[int] = Field(
        description="Optional: limit to the first n results. For descending sort this means n largest values, for ascending - n smallest values"
    )
