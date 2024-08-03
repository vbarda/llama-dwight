# NOTE: this is a global in-memory that's shared across agents
from typing import Any, Optional

TOOLKIT_KEY = "toolkit"


class InMemoryStore:
    store = {}

    def get(self, key: str) -> Optional[Any]:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value


IN_MEMORY_STORE = InMemoryStore()
