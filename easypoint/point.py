from __future__ import annotations

from math import cos, hypot, sin, radians as deg_to_rad
from typing import (
    Any,
    Callable,
    Iterator,
    Sequence,
    cast,
    overload,
    reveal_type,
)
from typing_extensions import Self


def applier(func: Merger) -> ApplyFunc:
    return lambda s, o: Point.from_(s).apply(Point.from_(o), func)


class Point(Sequence[float]):
    name: str = ""

    def __init__(self, *coords: float, loop: bool = False):
        self.coords = list(coords)
        self.loop = loop

    def named(self, new_name: str) -> Point:
        new: Point = self.copy()
        new.name = new_name
        return new

    @staticmethod
    def from_(value: PointLike, loop: bool = False) -> Point:
        if isinstance(value, Point):
            return value
        if isinstance(value, (list, tuple)):
            return Point(*value, loop=loop)
        return Point(value, loop=True)

    @overload
    def __getitem__(self, item: int) -> float:
        ...

    @overload
    def __getitem__(self, item: slice) -> Point:
        ...

    @overload
    def __getitem__(self, item: tuple[int, ...]) -> Point:
        ...

    @overload
    def __getitem__(self, item: int | slice | tuple[int, ...]) -> float | Point:
        ...

    def __getitem__(self, item: int | slice | tuple[int, ...]) -> float | Point:
        if isinstance(item, slice):  # type: ignore
            iterable = slice_iter(item, len(self.coords) if not self.loop else None)
            return Point(*(self[x] for x in iterable))

        if isinstance(item, tuple):
            return Point(*(self[x] for x in item))

        return self.get(item)

    def get(self, index: int) -> float:
        if self.loop:
            index = index % len(self)

        if index < 0:
            index = len(self) + index

        if index not in range(0, len(self)):
            return 0

        return self.coords[index]

    def concrete(self, loop: bool | None = None) -> Point:
        len(self)  # this hack raises error on infinite points
        return Point(*self.coords, loop=get_default(loop, self.loop))

    @overload
    def __setitem__(self, item: int, value: float):
        ...

    @overload
    def __setitem__(self, item: slice | tuple[int, ...], value: Sequence[float]):
        ...

    def __setitem__(
        self, item: int | slice | tuple[int, ...], value: float | Sequence[float]
    ):
        if isinstance(item, int):
            assert isinstance(value, (int, float))
            self.set([item], [value])
            return

        assert not isinstance(value, (int, float))

        if isinstance(item, slice):  # type: ignore
            iterable = slice_iter(item, len(self.coords) if not self.loop else None)
            self.set([*iterable], value)
            return

        self.set(item, value)

    def set(self, indices: Sequence[int], values: Sequence[float]):
        for index, value in zip(indices, values):
            self.coords[index] = value

    @property
    def x(self) -> float:
        return self[0]

    @property
    def y(self) -> float:
        return self[1]

    @property
    def z(self) -> float:
        return self[2]

    def apply(self, other: PointLike, func: Merger) -> Point:
        if isinstance(other, FnPoint):
            return other.apply(self, lambda s, o: func(o, s), rev=True)

        other = Point.from_(other)

        max_len = max(len(self), len(other))

        results: list[float] = []

        for i in range(max_len):
            results.append(func(self[i], other[i]))

        return Point(*results, loop=self.loop and other.loop)

    __add__ = applier(lambda s, o: s + o)
    __radd__ = applier(lambda s, o: o + s)
    __sub__ = applier(lambda s, o: s - o)
    __rsub__ = applier(lambda s, o: o - s)
    __mul__ = applier(lambda s, o: s * o)
    __rmul__ = applier(lambda s, o: o * s)
    __truediv__ = applier(lambda s, o: s / o)
    __rtruediv__ = applier(lambda s, o: o / s)
    __pow__ = applier(lambda s, o: cast(float, s**o))
    __rpow__ = applier(lambda s, o: cast(float, o**s))
    __mod__ = applier(lambda s, o: s % o)
    __rmod__ = applier(lambda s, o: o % s)
    __floordiv__ = applier(lambda s, o: s // o)
    __rfloordiv__ = applier(lambda s, o: o // s)

    def __round__(self, ndigits: int = 0) -> Point:
        return self.round(ndigits)

    def round(self, ndigits: int = 0) -> Point:
        return Point(
            round(self.x, ndigits),
            round(self.y, ndigits),
        )

    def __iter__(self) -> Iterator[float]:
        i = 0
        length = len(self)
        while True:
            if not self.loop and i == length:
                return
            yield self.get(i % length)
            i += 1

    def int_iter(self) -> Iterator[int]:
        return map(int, self)

    def transform(self, matrix: Matrix) -> Point:
        """
        Transforms a vecN using a given NxN matrix.
        """
        self_matrix = Matrix.as_matrix(self).transpose()

        product = matrix * self_matrix

        values = [cast(list[float], row)[0] for row in product.to_list()]

        return Point(*values)

    @overload
    def rotate2d(
        self, center: PointLike = 0, *, degrees: float, radians: None = None
    ) -> Point:
        ...

    @overload
    def rotate2d(
        self, center: PointLike = 0, *, degrees: None = None, radians: float
    ) -> Point:
        ...

    def rotate2d(
        self,
        center: PointLike = 0,
        *,
        degrees: float | None = None,
        radians: float | None = None,
        axis: tuple[int, int] = (0, 1),
    ) -> Point:
        error = ValueError(
            "Either degrees or radians should be provided, not both nor neither"
        )

        center = Point.from_(center)

        if radians is None:
            if degrees is None:
                raise error
            radians = deg_to_rad(degrees)
        elif degrees is not None:
            raise error

        diff = self - center

        mask: Point

        if isinstance(self, FnPoint):
            mask = FnPoint(lambda i: i in axis)
        else:
            mask = Point.from_([(i in axis) for i in range(len(self))])

        diff2d = diff * mask
        diffNd = diff - diff2d

        diff2d_norm = diff2d[axis]

        sine = sin(radians)
        cosine = cos(radians)

        rotation = Matrix.as_matrix((cosine, -sine), (sine, cosine))

        rotated = diff2d_norm.transform(rotation)

        diff2d[axis] = rotated

        return diff2d + center + diffNd

    def distance(self, other: PointLike = 0) -> float:
        return hypot(*(self - other))

    def normalized(self) -> Point:
        return self / self.distance()

    def __repr__(self) -> str:
        return f"{self._header()}[{self._content_repr()}]"

    def _header(self) -> str:
        cls_name = self.__class__.__name__
        if self.name:
            return f"{cls_name}<{self.name}>"
        return cls_name

    def _content_repr(self) -> str:
        points = ", ".join(map(str, self))
        if self.loop:
            return f"{points}, ..."
        return points

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (Point, float, int, list, tuple)):
            return False

        other = Point.from_(other)
        return eq((self - other).distance(), 0)

    def __len__(self):
        return len(self.coords)

    def copy(self) -> Point:
        return self.apply(0, lambda x, _: x)

    def interpolate(self, other: PointLike, k: float) -> Point:
        other = Point.from_(other)
        return other * k + self * (1 - k)

    def center(self, other: PointLike) -> Point:
        return self.interpolate(other, 0.5)


class FnPoint(Point):
    def __init__(
        self,
        func: Callable[[int], float],
        length: int | None = None,
    ):
        self.length = length
        self.func = func

    def derive(self, fn: Callable[[int], int]) -> FnPoint:
        return FnPoint(lambda i: self.get(fn(i)))

    def get(self, index: int | None) -> float:
        if index is None:
            return 0

        if self.length:
            if index not in range(-self.length, self.length):
                return 0
        return self.func(index)

    def set(self, indices: Sequence[int], values: Sequence[float]):
        data = {indices[i]: values[i % len(values)] for i in range(len(indices))}
        old_func = self.func

        def get_value(index: int):
            if index in data:
                return data[index]
            return old_func(index)

        self.func = get_value

    def concrete(self, loop: bool | None = None) -> Point:
        len(self)
        return Point(*self, loop=get_default(loop, False))

    @overload
    def __getitem__(self, item: int) -> float:
        ...

    @overload
    def __getitem__(self, item: slice | tuple[int, ...]) -> FnPoint:
        ...

    @overload
    def __getitem__(self, item: int | slice | tuple[int, ...]) -> float | FnPoint:
        ...

    def __getitem__(self, item: int | slice | tuple[int, ...]) -> float | FnPoint:
        if isinstance(item, int):
            return self.get(item)

        if isinstance(item, tuple):

            def new_get(index: int | None) -> float:
                if index is None:
                    return 0
                return self.get(item[index])

            return FnPoint(new_get, length=len(item))

        func, length = calc_slice_func(item, self.length, self.get)

        return FnPoint(func, length)

    @property
    def loop(self):
        return self.length is None

    def __len__(self):
        if self.length is not None:
            return self.length

        raise ValueError("This Point doesn't have a length")

    def __eq__(self, other: object) -> bool:
        return self is other

    def apply(self, other: PointLike, func: Merger, rev: bool = False) -> FnPoint:
        other = Point.from_(other)

        def new_func(index: int) -> float:
            return func(self.get(index), other.get(index))

        new_length = self.length if not rev else len(other)

        return FnPoint(new_func, new_length)

    def __iter__(self):
        i = 0
        while True:
            if i == self.length:
                return
            yield self.get(i)
            i += 1

    def _content_repr(self):
        if self.length:
            return f"..., elements: {self.length}"
        return "..."

    def copy(self):
        return FnPoint(self.func, self.length)


from .internal_types import ApplyFunc, Merger, PointLike
from .matrix import Matrix
from .utils import calc_slice_func, eq, get_default, slice_iter
