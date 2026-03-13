from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Iterable, List, Tuple, TypeVar

K = TypeVar("K")
V = TypeVar("V")
Q = TypeVar("Q")


class BaseIndex(ABC, Generic[K, V, Q]):
    """Common interface shared by local indexes."""

    @abstractmethod
    def build(self, items: Iterable[Tuple[K, V]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def insert(self, key: K, value: V) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(self, q: Q) -> List[V]:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__


class DistributedIndex(ABC):
    """Small interface for distributed indexes so the experiment layer can stay generic."""

    @abstractmethod
    def build(self, records_rdd) -> float:
        raise NotImplementedError

    @abstractmethod
    def point_query(self, key):
        raise NotImplementedError

    @abstractmethod
    def range_query(self, low, high):
        raise NotImplementedError
