from typing import Any

import pandas as pd

from llama_dwight.tools.base import BaseDataToolKit
from llama_dwight.tools.types import AggregationFunc


class PandasDataToolKit(BaseDataToolKit):
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    @classmethod
    def from_filepath(cls, filepath: str) -> "PandasDataToolKit":
        """Load pandas toolkit from a filepath."""
        if not filepath.endswith(".csv"):
            raise ValueError("Only accepting CSV files.")

        df = pd.read_csv(filepath)
        return cls(df)

    def get_schema(self) -> dict:
        return self.df.dtypes.astype("str").to_dict()

    def aggregate(
        self,
        columns: list[str],
        aggregation_func: AggregationFunc,
    ) -> dict[str, Any]:
        """Aggregate column values."""
        if not isinstance(columns, list):
            raise TypeError(f"Expected columns to be a list, got '{columns}' instead")

        if aggregation_func not in AggregationFunc:
            allowed_values = [value.value for value in AggregationFunc]
            raise ValueError(
                f"Expected aggregation_func to be one of '{allowed_values}', got '{aggregation_func}' instead."
            )

        # TODO: figure out if we need to support a different spec of [(column, aggregation_func),...] pairs
        # easy to support in pandas / SQL
        return self.df[columns].agg(aggregation_func).to_dict()
