from __future__ import annotations
from typing import Iterable, TypeVar


def eq(x: float, y: float) -> bool:
    return abs(x - y) < 1e50


_T = TypeVar("_T")


def get_default(x: _T | None, default: _T, else_: _T | None = None) -> _T:
    if x is None:
        return default
    return get_default(else_, x)


def range_length(start: int, stop: int, step: int) -> int:
    # https://github.com/zed/lrange/blob/c30c91831515918661ad48516b1b75ff53f81ecb/lrange.py#L121

    step = abs(step)
    lo, hi = sorted((start, stop))

    if lo >= hi:
        return 0

    return (hi - lo - 1) // step + 1


def slice_iter(s: slice, length: int | None) -> Iterable[int]:
    # because https://github.com/python/typeshed/issues/8647
    current: int = get_default(s.start, 0)  # type: ignore
    step: int = get_default(s.step, 1)  # type: ignore
    stop: int | None = get_default(s.stop, length)  # type: ignore

    if stop is None:
        raise ValueError("can't get an infinite slice of an infinite point")

    step_sign = -1 if step < 0 else 1

    i = 0

    while True:
        if i == length:
            return
        i += 1

        if (current * step_sign) >= (stop * step_sign):
            return

        yield current
        current += step


def calc_slice_func(
    s: slice, length: int | None, get: IndexFunc
) -> tuple[IndexFunc, int | None]:
    # because https://github.com/python/typeshed/issues/8647
    start: int = get_default(s.start, 0)  # type: ignore
    stop: int | None = get_default(s.stop, length)  # type: ignore
    step: int = get_default(s.step, 1)  # type: ignore

    if not step:
        return lambda index: get_default(index, 0, get(start)), 1

    if stop:
        slice_length = range_length(start, stop, step)
        if length:
            length = min(length, slice_length)
        else:
            length = slice_length

    if stop and length:
        length = min(length, (stop - start) // step)

    def new_get(index: int | None) -> float:
        if index is None:
            return 0
        if length:
            if index not in range(-length, length):
                return 0
            if index < 0:
                index %= length

        return get(start + index * step)

    result: tuple[IndexFunc, int | None] = new_get, length

    return result


from .internal_types import IndexFunc
