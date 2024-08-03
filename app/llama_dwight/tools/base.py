import abc
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from llama_dwight.tools.types import AggregationFunc, AggregationInput, GroupbyInput


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

    def get_schema(self) -> dict:
        """Get schema associated with the data toolkit."""
        raise NotImplementedError

    def get_tools(self) -> list[BaseTool]:
        return [
            StructuredTool(
                name="aggregate",
                description='Aggregate column values. DO NOT use this if asked for a group by aggregation. Example: "what was the total sales amount?"',
                func=self.aggregate,
                args_schema=AggregationInput,
            ),
            StructuredTool(
                name="groupby",
                description='Group by a list of columns and calculate aggregated value for each group. Example: "what was the total sales amount?"',
                func=self.groupby,
                args_schema=GroupbyInput,
            ),
        ]
