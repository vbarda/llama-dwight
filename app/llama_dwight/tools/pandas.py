from typing import Any

import pandas as pd

from llama_dwight.tools.base import BaseDataToolKit
from llama_dwight.tools.types import AggregationFunc, validate_aggregation_func


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
        return self.df.dtypes.astype("str").str.replace("object", "str").to_dict()

    def aggregate(
        self,
        columns: list[str],
        aggregation_func: AggregationFunc,
    ) -> dict[str, Any]:
        """Aggregate column values."""
        if not isinstance(columns, list):
            raise TypeError(f"Expected columns to be a list, got '{columns}' instead")

        validate_aggregation_func(aggregation_func)

        # TODO: figure out if we need to support a different spec of [(column, aggregation_func),...] pairs
        # easy to support in pandas / SQL
        return self.df[columns].agg(aggregation_func).to_dict()

    def groupby(
        self,
        groupby_columns: list[str],
        value_column: str,
        aggregation_func: AggregationFunc,
    ) -> dict[tuple[str, ...], Any]:
        if not isinstance(groupby_columns, list):
            raise TypeError(
                f"Expected groupby_columns to be a list, got '{groupby_columns}' instead"
            )

        validate_aggregation_func(aggregation_func)
        return (
            self.df.groupby(groupby_columns)[value_column]
            .agg(aggregation_func)
            .to_dict()
        )
