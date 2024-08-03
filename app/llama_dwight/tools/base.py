import abc
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from llama_dwight.tools.types import AggregationFunc, AggregationInput


class BaseDataToolKit(abc.ABC):
    def aggregate(
        self,
        columns: list[str],
        aggregation_func: AggregationFunc,
    ) -> dict[str, Any]:
        """Aggregate column values.

        Args:
            columns: List of columns to perform aggregation on
            aggregation_func: Aggregation function to apply to the columns. REMEMBER: Average always refers to 'mean'
        """
        raise NotImplementedError

    def get_schema(self) -> dict:
        """Get schema associated with the data toolkit."""
        raise NotImplementedError

    def get_tools(self) -> list[BaseTool]:
        return [
            StructuredTool(
                name="aggregate",
                description="Aggregate column values.",
                func=self.aggregate,
                args_schema=AggregationInput,
            )
        ]
