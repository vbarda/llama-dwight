import abc
from typing import Any, Optional

from langchain_core.tools import BaseTool, StructuredTool

from llama_dwight.tools.types import (
    AggregationFunc,
    AggregationInput,
    GroupbyInput,
    FilterSpec,
    FilterInput,
    SortInput,
    ToolName,
)


class BaseDataToolKit(abc.ABC):
    def aggregate(
        self,
        columns: list[str],
        aggregation_func: AggregationFunc,
    ) -> dict[str, Any]:
        """Aggregate column values. See AggregationInput for args description."""
        raise NotImplementedError

    def groupby(
        self,
        groupby_columns: list[str],
        value_column: str,
        aggregation_func: AggregationFunc,
    ) -> dict[tuple[str, ...], Any]:
        """Group by a list of columns and calculate aggregated value for each group. See GroupbyInput for args description."""
        raise NotImplementedError

    def filter(self, filters: list[FilterSpec]) -> None:
        """Filter the data."""
        raise NotImplementedError

    def sort(self, column: str, ascending: bool, limit: Optional[int]) -> None:
        """Sort the data or find top/bottom n values."""
        raise NotImplementedError

    def get_schema(self) -> dict:
        """Get schema associated with the data toolkit."""
        raise NotImplementedError

    def get_tools(self) -> list[BaseTool]:
        return [
            StructuredTool(
                name=ToolName.FILTER,
                description='Filter dataset using a list of filter specifications. Example: "transactions greater than 10"',
                func=self.filter,
                args_schema=FilterInput,
            ),
            StructuredTool(
                name=ToolName.SORT,
                description='Sort dataset and optionally find top/botton n values. Example: "largest companies" / "bottom 5 cities by population"',
                func=self.sort,
                args_schema=SortInput,
            ),
            StructuredTool(
                name=ToolName.AGGREGATE,
                description='Aggregate column values. DO NOT use this if asked for a group by aggregation. Example: "what was the total sales amount?"',
                func=self.aggregate,
                args_schema=AggregationInput,
            ),
            StructuredTool(
                name=ToolName.GROUPBY,
                description='Group by a list of columns and calculate aggregated value for each group. Example: "what was the total sales amount?"',
                func=self.groupby,
                args_schema=GroupbyInput,
            ),
        ]

    def clear(self) -> None:
        """Clear any intermediate toolkit state."""
        raise NotImplementedError
