from __future__ import annotations
from typing import Callable, Iterator, Union


def _i_to_index(i: int, size: Size) -> Index:
    result_base = []

    # x y z
    for di, dimension in enumerate(size[::-1]):
        if di != len(size) - 1:
            result_base.append(i % dimension)
        else:
            result_base.append(i)
            break
        i //= dimension

    return tuple(result_base[::-1])


def _index_to_i(index: Index, size: Size) -> int:
    result = 0
    dim_size = 1

    rev_index = index[::-1]

    for i, (index_dimension, size_dimension) in enumerate(zip(rev_index, size[::-1])):
        if i != len(index) - 1:
            index_dimension %= size_dimension
        result += dim_size * index_dimension
        dim_size *= size_dimension

    return result


class MatrixIter:
    def __init__(
        self, matrix: Matrix, start: Index | None = None, stop: Index | None = None
    ):
        self.matrix = matrix
        self.index = _index_to_i(start, matrix.size) if start else 0
        self.stop = stop or _i_to_index(
            1 + _index_to_i(self.matrix.last_index, matrix.size), matrix.size
        )

    def __iter__(self):
        return self

    def __next__(self) -> Index:
        index = _i_to_index(self.index, self.matrix.size)
        if index == self.stop:
            raise StopIteration
        self.index += 1
        return index


class Matrix:
    def __init__(self, size: Size, default_value: int = 0):
        self.data: dict[int, float] = {}
        self.size = size
        self.default_value = default_value

    @property
    def last_index(self) -> Index:
        return tuple(x - 1 for x in self.size)

    @property
    def _flat_size(self) -> int:
        return _index_to_i(self.last_index, self.size) + 1

    def cut(self, index: Index) -> Matrix:
        new_size = tuple(x - 1 for x in self.size)
        new = Matrix(new_size)

        for cur_index in self:
            shifted_index = tuple(x - (x > index[i]) for i, x in enumerate(cur_index))
            new[shifted_index] = self[cur_index]

        return new

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix):
            return NotImplemented

        return (
            self.size == other.size
            and (self.default_value == other.default_value)
            and all(self.get_i(i) == other.get_i(i) for i in range(self._flat_size))
        )

    def get_i(self, i: int) -> float:
        return self.data.get(i, self.default_value)

    def set_i(self, i: int, value: float = 0) -> None:
        if value == self.default_value:
            self.data.pop(i, 0)
            return
        self.data[i] = value

    def get_index(self, index: Index) -> float:
        i = _index_to_i(index, self.size)
        return self.get_i(i)

    def set_index(self, index: Index, value: float) -> None:
        i = _index_to_i(index, self.size)
        return self.set_i(i, value)

    def __getitem__(self, index: Index) -> float:
        return self.get_index(index)

    def __setitem__(self, index: Index, value: float) -> None:
        return self.set_index(index, value)

    def __iter__(self) -> Iterator[Index]:
        return self.iter()

    def iter(
        self, start: Index | None = None, stop: Index | None = None
    ) -> Iterator[Index]:
        return MatrixIter(self, start, stop)

    def __str__(self) -> str:
        return f"Matrix object ({'x'.join(map(str, self.size))} matrix)"

    def __matmul__(self, other: Matrix) -> Matrix:
        return self * other

    def __mul__(self, other: Union[Matrix, float]) -> Matrix:
        if isinstance(other, (int, float)):
            new = self.new()
            for index in self:
                new[index] = self[index] * other
            return new

        if not (self.size and other.size):
            raise ValueError("Can't multiply empty matrices")

        if self.size[-1] != other.size[0]:
            raise ValueError(
                f"Can't multiply {self} and {other}, since {self.size[-1]} != {other.size[0]}"
            )

        new_size = (*self.size[:-1], *other.size[1:])
        new = Matrix(new_size)
        split = len(self.size) - 1

        for index in new:
            result: float = 0
            for x in range(self.size[split]):
                s = self[*index[:split], x]
                o = other[x, *index[split:]]
                result += s * o

            new[index] = result
        return new

    def __add__(self, other: Matrix) -> Matrix:
        return self.apply_bin(
            other,
            lambda s, o: s + o,
            "-",
        )

    def __sub__(self, other: Matrix) -> Matrix:
        return self.apply_bin(
            other,
            lambda s, o: s - o,
            "-",
        )

    def __neg__(self) -> Matrix:
        return self.apply(lambda x: -x)

    def new(self) -> Matrix:
        return Matrix(self.size)

    def copy(self) -> Matrix:
        return self.apply(lambda x: x)

    def apply(self, func: Callable[[float], float]) -> Matrix:
        new = self.new()
        for index in new:
            new[index] = func(self[index])
        return new

    def apply_bin(
        self, other: Matrix, func: Callable[[float, float], float], op: str = "?"
    ) -> Matrix:
        if self.size != other.size:
            raise ValueError(
                f"Can't apply '{op}' to {self} and {other} (different sizes)"
            )

        new = self.new()
        for index in new:
            new[index] = func(self[index], other[index])
        return new

    def get_submatrix(self, i: int) -> Matrix:
        zeros = (0,) * (len(self.size) - 1)
        new_size = self.size[1:]
        submatrix = Matrix(new_size, default_value=self.default_value)

        control = None

        for control, *rest in self.iter((i, *zeros)):
            new_index = tuple(rest)  # `rest` is a list

            if control != i:
                return submatrix

            submatrix[new_index] = self[control, *new_index]

        if control is not None:
            return submatrix

        raise ValueError("can't get a submatrix for an empty matrix")

    def to_list(self) -> MatrixList:
        result: MatrixList = []
        if len(self.size) == 1:
            return [self[index] for index in self]

        for i in range(self.size[0]):
            result.append(self.get_submatrix(i).to_list())

        return result

    def transpose(self) -> Matrix:
        new = Matrix(self.size[::-1])
        for index in new:
            new[index] = self[index[::-1]]
        return new

    @staticmethod
    def scalar(size: int, dimensions: int = 2, num: float = 1) -> Matrix:
        new = Matrix((size,) * dimensions)
        for i in range(size):
            new[(i,) * dimensions] = num
        return new

    @staticmethod
    def as_matrix(*points: PointLike) -> Matrix:
        data: list[list[float]] = []
        max_len = 0

        for point in points:
            data.append([*Point.from_(point)])
            max_len = max(len(data[-1]), max_len)

        result: Matrix = Matrix((len(points), max_len))

        for i, values in enumerate(data):
            for j, value in enumerate(values):
                result[i, j] = value

        return result


from .internal_types import Index, MatrixList, PointLike, Size
from .point import Point
