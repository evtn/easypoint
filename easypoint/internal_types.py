from __future__ import annotations
from typing import Callable, List, Tuple, Union


PointLike = Union["Point", float, List[float], Tuple[float, ...]]

Index = Tuple[int, ...]
Size = Index

MatrixList = List[Union[float, "MatrixList"]]

Merger = Callable[[float, float], float]
ApplyFunc = Callable[[PointLike, PointLike], "Point"]
IndexFunc = Callable[[Union[int, None]], float]
MatrixIndexFunc = Callable[[Index], float]

from .point import Point
