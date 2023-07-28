from typing import Callable, Union


PointLike = Union["Point", float, list[float], tuple[float, ...]]

Index = tuple[int, ...]
Size = Index

MatrixList = list[Union[float, "MatrixList"]]

Merger = Callable[[float, float], float]
ApplyFunc = Callable[[PointLike, PointLike], "Point"]
IndexFunc = Callable[[int | None], float]
MatrixIndexFunc = Callable[[Index], float]

from .point import Point
