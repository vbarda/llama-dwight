from typing import Any, Union, Optional

from sqlalchemy import Engine, create_engine, text as sql_text

from llama_dwight.tools.types import FilterSpec, FilterOperator, FilterValueType
from llama_dwight.tools.base import BaseDataToolKit
from llama_dwight.tools.types import AggregationFunc, validate_aggregation_func


VIEW_PREFIX = "result"


def get_sql_aggregation_operator(aggregation_func: AggregationFunc) -> None:
    aggregation_func_to_sql_operator = {
        AggregationFunc.SUM: "SUM",
        AggregationFunc.MEAN: "AVG",
        AggregationFunc.MIN: "MIN",
        AggregationFunc.MAX: "MAX",
        AggregationFunc.COUNT: "COUNT",
    }
    if aggregation_func not in aggregation_func_to_sql_operator:
        raise ValueError(
            f"Aggregation '{aggregation_func}' is not supported for SQL currently."
        )
    return aggregation_func_to_sql_operator[aggregation_func]


def convert_filter_value(value: str, value_type: FilterValueType) -> Union[str, float]:
    if value_type in {FilterValueType.STRING, FilterValueType.DATETIME}:
        return f"'{value}'"
    elif value_type == FilterValueType.NUMBER:
        return float(value)
    else:
        raise ValueError(f"Unsupported value type '{value_type}'")


class SQLDataToolKit(BaseDataToolKit):
    def __init__(self, engine: Engine, table_name: str) -> None:
        self.engine = engine
        self.table_name = table_name
        self.views = []
        self.create_view(f"SELECT * FROM {self.table_name}")

    def create_view(self, query: str) -> None:
        view_name = f"{VIEW_PREFIX}_{len(self.views)}"
        drop_query = sql_text(f"DROP VIEW IF EXISTS {view_name}")
        create_query = sql_text(f"CREATE VIEW {view_name} AS {query}")
        with self.engine.connect() as conn:
            conn.execute(drop_query)
            conn.execute(create_query)

        self.views.append(view_name)

    @property
    def current_view_name(self) -> Optional[str]:
        if len(self.views) == 0:
            return None

        return self.views[-1]

    @classmethod
    def from_conn_string(cls, conn_string: str, table_name: str) -> "SQLDataToolKit":
        """Load DB from connection and table."""
        if "sqlite" not in conn_string:
            raise ValueError("Only SQLite DB is supported at the moment.")

        engine = create_engine(conn_string)
        return cls(engine, table_name)

    def get_schema(self) -> dict:
        with self.engine.connect() as conn:
            cur = conn.execute(sql_text(f"PRAGMA table_info({self.table_name});"))
            columns = cur.keys()
            res = cur.fetchall()

        schema_info = [dict(zip(columns, r)) for r in res]
        return {field_info["name"]: field_info["type"] for field_info in schema_info}

    def aggregate(
        self,
        columns: list[str],
        aggregation_func: AggregationFunc,
    ) -> dict[str, Any]:
        """Aggregate column values."""
        if not isinstance(columns, list):
            raise TypeError(f"Expected columns to be a list, got '{columns}' instead")

        aggregation_operator = get_sql_aggregation_operator(aggregation_func)
        # NOTE: we preserve the original column names for simplicity
        aggregations = [f"{aggregation_operator}({col}) AS {col}" for col in columns]
        aggregation = ", ".join(aggregations)
        self.create_view(f"SELECT {aggregation} FROM {self.current_view_name}")
        with self.engine.connect() as conn:
            # NOTE: at this point current view is the latest
            res = conn.execute(
                sql_text(f"SELECT * FROM {self.current_view_name}")
            ).fetchall()
        return dict(zip(columns, res[0]))

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

        if freq is not None:
            raise ValueError("Grouping on date columns is not supported")

        aggregation_operator = get_sql_aggregation_operator(aggregation_func)
        # NOTE: we preserve the original column name for simplicity
        aggregation = f"{aggregation_operator}({value_column}) AS {value_column}"
        groupby = ", ".join(groupby_columns)
        self.create_view(
            f"SELECT {aggregation}, {groupby} FROM {self.current_view_name} GROUP BY {groupby}"
        )
        with self.engine.connect() as conn:
            # NOTE: at this point current view is the latest
            cur = conn.execute(sql_text(f"SELECT * FROM {self.current_view_name}"))
            columns = cur.keys()
            res = cur.fetchall()
        return [dict(zip(columns, r)) for r in res]

    def filter(self, filters: list[FilterSpec]) -> None:
        if not filters:
            return

        wheres = []
        filter_operator_to_sql_operator = {
            FilterOperator.EQ: "=",
            FilterOperator.NEQ: "<>",
            FilterOperator.GREATER: ">",
            FilterOperator.GREATER_OR_EQ: ">=",
            FilterOperator.LESS: "<",
            FilterOperator.LESS_OR_EQ: "<=",
        }
        for filter_spec in filters:
            value = convert_filter_value(filter_spec.value, filter_spec.value_type)

            if filter_spec.operator not in filter_operator_to_sql_operator:
                raise ValueError(
                    f"Filter operator '{filter_spec.operator}' not supported."
                )

            sql_operator = filter_operator_to_sql_operator[filter_spec.operator]
            wheres.append(f"{filter_spec.column} {sql_operator} {value}")

        where = " AND ".join(wheres)
        self.create_view(f"SELECT * FROM {self.current_view_name} WHERE {where}")
        return "Successfully filtered data."

    def sort(self, column: str, ascending: bool, limit: int | None) -> None:
        sort_suffix = "ASC" if ascending else "DESC"
        sort_query = (
            f"SELECT * FROM {self.current_view_name} ORDER BY {column} {sort_suffix}"
        )
        if limit is None:
            self.create_view(sort_query)
        else:
            self.create_view(sort_query + f"LIMIT {limit}")

        if limit:
            with self.engine.connect() as conn:
                # NOTE: at this point current view is the latest
                cur = conn.execute(sql_text(f"SELECT * FROM {self.current_view_name}"))
                columns = cur.keys()
                res = cur.fetchall()
            return [dict(zip(columns, r)) for r in res]
        else:
            return "Successfully sorted data."

    def clear(self) -> None:
        while self.views:
            view_name = self.views.pop()
            with self.engine.connect() as conn:
                conn.execute(sql_text(f"DROP VIEW IF EXISTS {view_name}"))
