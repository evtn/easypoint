from __future__ import annotations
from typing import Callable, List, Optional, Tuple, Union

BasePointLike = Union[float, List[float], Tuple[float, ...]]
PointLike = Union["Point", BasePointLike]

Index = Tuple[int, ...]
Size = Index

MatrixList = List[Union[float, "MatrixList"]]

Merger = Callable[[float, float], float]
ApplyFunc = Callable[[PointLike, PointLike], "Point"]
IndexFunc = Callable[[Optional[int]], float]
MatrixIndexFunc = Callable[[Index], float]

from .point import Point
