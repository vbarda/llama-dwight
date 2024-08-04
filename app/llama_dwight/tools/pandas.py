from typing import Any, Union, Optional

import pandas as pd

from llama_dwight.tools.types import FilterSpec, FilterOperator, FilterValueType
from llama_dwight.tools.base import BaseDataToolKit
from llama_dwight.tools.types import AggregationFunc, validate_aggregation_func


def convert_filter_value(value: str, value_type: FilterValueType) -> Union[str, float]:
    if value_type == FilterValueType.STRING:
        return value
    elif value_type == FilterValueType.NUMBER:
        return float(value)
    elif value_type == FilterValueType.DATETIME:
        return pd.to_datetime(value)
    else:
        raise ValueError(f"Unsupported value type '{value_type}'")


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    for column_name in df.columns:
        if "date" in column_name.lower():
            try:
                df[column_name] = pd.to_datetime(df[column_name])
            except ValueError:
                continue


class PandasDataToolKit(BaseDataToolKit):
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        # this will be used for any intermediate outputs of the latest tool call
        # such as filter, sort etc.
        self.current_df = df.copy()

    @classmethod
    def from_filepath(
        cls, filepath: str, preprocess: bool = False
    ) -> "PandasDataToolKit":
        """Load pandas toolkit from a filepath."""
        if not filepath.endswith(".csv"):
            raise ValueError("Only accepting CSV files.")

        df = pd.read_csv(filepath)
        if preprocess:
            preprocess_df(df)
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
        return self.current_df[columns].agg(aggregation_func).to_dict()

    def groupby(
        self,
        groupby_columns: list[str],
        value_column: str,
        aggregation_func: AggregationFunc,
        freq: Optional[str],
    ) -> dict[tuple[str, ...], Any]:
        if not isinstance(groupby_columns, list):
            raise TypeError(
                f"Expected groupby_columns to be a list, got '{groupby_columns}' instead"
            )

        validate_aggregation_func(aggregation_func)
        if freq is None:
            by = groupby_columns
        else:
            if len(groupby_columns) > 1:
                raise ValueError(
                    "Can only group by a single date column when frequency is specified"
                )

            by = pd.Grouper(key=groupby_columns[0], freq=freq)
        agg_df = self.current_df.groupby(by)[value_column].agg(aggregation_func)
        self.current_df = agg_df.reset_index()
        return agg_df.to_dict()

    def filter(self, filters: list[FilterSpec]) -> None:
        if not filters:
            return

        mask = None
        for filter_spec in filters:
            value_series = self.current_df[filter_spec.column]
            value = convert_filter_value(filter_spec.value, filter_spec.value_type)
            if filter_spec.operator == FilterOperator.EQ:
                new_mask = value_series == value
            elif filter_spec.operator == FilterOperator.NEQ:
                new_mask = value_series != value
            elif filter_spec.operator == FilterOperator.GREATER:
                new_mask = value_series > value
            elif filter_spec.operator == FilterOperator.GREATER_OR_EQ:
                new_mask = value_series >= value
            elif filter_spec.operator == FilterOperator.LESS:
                new_mask = value_series < value
            elif filter_spec.operator == FilterOperator.LESS_OR_EQ:
                new_mask = value_series <= value
            else:
                raise ValueError(
                    f"Filter operator '{filter_spec.operator}' not supported."
                )

            if mask is None:
                mask = new_mask
            else:
                mask = mask & new_mask

        # set intermediate outputs to filtered df
        self.current_df = self.current_df[mask]
        return "Successfully filtered data."

    def sort(self, column: str, ascending: bool, limit: int | None) -> None:
        if limit is None:
            self.current_df = self.current_df.sort_values(column, ascending=ascending)
        elif ascending:
            self.current_df = self.current_df.nsmallest(limit, column)
        else:
            self.current_df = self.current_df.nlargest(limit, column)

        if limit:
            return self.current_df.to_dict(orient="records")
        else:
            return "Successfully sorted data."

    def clear(self) -> None:
        self.current_df = self.df.copy()
