# NOTE: this is a global in-memory that's shared across agents
from typing import Any, Optional

import pandas as pd

DATAFRAME_KEY = "dataframe"


class InMemoryStore:
    store = {}

    def get(self, key: str) -> Optional[Any]:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value


def load_data(filepath: str):
    if not filepath.endswith(".csv"):
        raise ValueError("Only accepting CSV files.")

    df = pd.read_csv(filepath)
    IN_MEMORY_STORE.set(DATAFRAME_KEY, df)


IN_MEMORY_STORE = InMemoryStore()
