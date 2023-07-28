# Minimal general-purpose vector / matrix arithmetics library

# Introduction

easypoint has 2 main types to work with: `Point` (a.k.a. `Vector`) and `Matrix`

`Point` class builds up on my previous work with [evtn/soda](https://github.com/evtn/soda) and [evtn/soda-old](https://github.com/evtn/soda-old).    
Both being graphics-oriented, so vector arithmetics is a must-have. 

But over time, `Point` became a convenient class for various non-graphical tasks and tasks out of scope for `soda` (e.g. raster graphics).    
This module also brings a refined `Matrix` class I've been using in various private/unfinished projects (an old version can be seen [here](https://gist.github.com/evtn/8683e58770f2901527275d46465e4cbe))

Both are refined and generalized for N dimensions. Some new additions (like `Point.transform(matrix: Matrix)`) are also in place.

# Usage

## Point

Point/Vector (`easypoint.Vector` is just an alias) is a fancy `Sequence[float]`, supporting various convenient operations.

### Make a point

```python
from easypoint import Point

# create a Point with numbers:

p1 = Point(1, 2, 3) # Point[1, 2, 3]

# ...or from a list/tuple

p2 = Point.from_([1, 2, 3]) # Point[1, 2, 3]

# ...from a number

p3 = Point.from_(1) # Point[1, ...]

# ...from another Point

p4 = p1[:2] # Point[1, 2]
p5 = p1[0, 2, 1] # Point[1, 3, 2] 
p6 = p3[:] # Error (a slice of an infinite Point)
```

```python
a = Point(1, 2)
b = Point(4, 5)
```

In any context where a point could be used, "point-like" values can be used:

-   `(1, 2)` <-> `Point(1, 2)`
-   `[1, 2]` <-> `Point(1, 2)`
-   `1` <-> `Point(1, loop=True)`

### Math

You can perform mathematical operations on points (element-wise):

```python
a + b # Point[5, 7]
a - b # Point[-3, -3]
a * b # Point[4, 10]
a / b # Point[0.25, 0.4]
a % b # Point[1, 2]
```

...and any point-like values:

```python
a + 10 # Point[11, 12]
a * 2 # Point[2, 4]
```

### Distance

You also can calculate distance between points and get a normalized vector:

```python
from math import pi

a.distance(b) # 4.242640687119285
a.distance() # 2.23606797749979 (distance between a and (0, 0), basically the length of a vector)
a.normalized() # Point[0.4472135954999579, 0.8944271909999159]
```

### Rotation

2D Rotation can be done around some center:

```python
a.rotate2d(degrees=90) # Point[-2, 1]
a.rotate2d(center=(10, 10), radians=pi / 2) # Point[18, 1]
a.rotate2d(center=10, degrees=90) # Point[18, 1]

# if you want to use axis other than (0, 1), pass `axis`:
c = Point(4, 6, 2, 3, 2)
c.rotate2d(center=10, radians=pi / 2, axis=(3, 4)) # Point[4, 6, 2, 18, 3]
```

### Transforms

You can transform an N-dimensional `Point` with a NxN `Matrix`:

```python
from easypoint import Point, Matrix
# Shearing
# 1 k
# 0 1

matrix = Matrix.as_matrix((1, 4), (0, 1))
t = Point(0, 5)
t.transform(matrix) # Point[20, 5]
```

### Looped points

Sometimes it's convenient to have a point with `p[i] == p[i % n]` (a repeating set of coordinates).    
It can be achieved by passing `loop=True` into `Point` constructor or `Point.from_`:

```python
p1 = Point(10.3, loop=True) # Point[10.3, ...]
p1[54378] # 10.3

p2 = Point.from_([1, 2, 3], loop=True) # Point[1, 2, 3, ...]
p2[540:550] # Point[1, 2, 3, 1, 2, 3, 1, 2, 3, 1]
```

Keep in mind that `Point.from_(int)` always produces a looped point, if you need a 1-dimensional point, use `Point(int)`

### Indexing

Points support three types of indexing:

- `point[int]` returns a value at that index, or 0 if this index doesn't exist (and the point is not looped)
- `point[slice]` returns a `Point` with values under that slice
- `point[tuple[int, ...]]` returns a Point with values under indices in the tuple

```python
a = Point(*range(5)) # Point[0, 1, 2, 3, 4]
a[2] # 2
a[2:4] # Point[2, 3, 4]
a[4, 3, 8, 2] # Point[4, 3, 0, 2]
```

Same applies for setting values on indices.   
Keep in mind that setting a slice/tuple doesn't change the dimension count, extra indices/values are ignored

There are also `x`, `y`, and `z` properties as aliases for `[0]`, `[1]`, and `[2]`

### Interpolation

For convenience, there are `point.interpolate(other: PointLike, k: float)` to interpolate between two points (self at 0, other at 1).    
`point.center(other: PointLike)` is an alias for `point.interpolate(other, 0.5)`

### Naming

You can give any point a name (any string) for convenience and better output:

```python
a = Point(3, 4) # Point[3, 4]
b = a.named("B") # Point<B>[3, 4]
```

Naming returns a copy of the point, so the original one is not renamed


### FnPoint

FnPoint is a Point defined by index function:

```python
from easypoint import FnPoint

fp = FnPoint(lambda i: 126 * i * i + 7)
fp[4] # 2023
```

...and optional length:

```python
from easypoint import FnPoint

fp = FnPoint(lambda i: 126 * i * i + 7, length=3)
fp[4] # 0

```

It is fully compatible with Point, but any operation on FnPoint will return you a new, derived FnPoint.    

If you want (for some reason) to get a concrete `Point` instance, call `fp.concrete(loop: bool = False)`    
Obviously, this will raise an error on an infinite FnPoint, so either pass a length into the constructor or as a slice:

```python

fp = FnPoint(lambda x: x * 2) # infinite point
fp_fin = FnPoint(lambda x: x * 2, length=4) # finite point
fp_slice = fp[:4] # also finite

# okay
fp_fin.concrete()
fp_slice.concrete()

# error
fp.concrete() 

```

## Matrix

Now you can wake up and take a non-pointy pill, at last.    

Matrices are N-dimensional tables, well, you can [read Wikipedia](https://en.wikipedia.org/wiki/Matrix_(mathematics)) instead of this.

In `easypoint`, matrices are quite straightforward (keep in mind, they have 0-based indexing):

```python
from easypoint import Matrix


mul_table = Matrix((10, 10))

for (y, x) in mul_table:
    mul_table[y, x] = (y + 1) * x

from pprint import pprint

"""
[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 ...,
 [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]]
"""
pprint(mul_table.to_list())
```

As you can see, one can easily iterate over every index in matrix using iterator protocol. If you want to iterate over some portion of a matrix, use `matrix.iter()` explicitly:

```python
# Matrix.iter(self, start: Index | None = None, stop: Index | None = None):

mul_table = Matrix((10, 10))

for index in mul_table.iter((3, 3), (4, 5)):
    mul_table[index] = 0 # why? idk

```

### Operations

As with `Point`, with matrices you can get an element-wise sum, difference and multiply matrix by a number.    
Multiplication (as well as `@`) is reserved for matrix multiplication (or, generally, tensor contraction).    
If you need an element-wise multiplication (or any other operation), you can use `Matrix.apply_bin`:

```python
from easypoint import Matrix

x_table = Matrix((10, 10))
y_table = x_table.new() # creates a new matrix of the same size

for (y, x) in x_table:
    x_table[y, x] = (x + 1)
    y_table[y, x] = (y + 1)

# Matrix.apply_bin(self, other: Matrix, func: Callable[[float, float], float], op: str = "?")
# `op` param is optional, it is an arbitrary string used for better debug
mul_table = x_table.apply_bin(y_table, lambda x, y: x * y, op="*")

from pprint import pprint

"""
[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 ...,
 [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]]
"""
pprint(mul_table.to_list())

```

You can also apply a function to a single matrix with `apply`:

```python
coord_table = Matrix((10, 10))

for (y, x) in coord_table:
    coord_table[y, x] = (x + y)

for (y, x) in coord_table:
    coord_table.apply(lambda x: -x) # same as coord_table * -1

```

Other methods defined:

- `matrix.new()` creates an empty matrix of the same size (same as `Matrix(matrix.size)`),
- `matrix.copy()` copies the matrix (same as `matrix.apply(x: x)`)
- `matrix.transpose()` transposes the matrix (wow!)
- `matrix.cut(index)` returns a new matrix where all the rows/columns/etc. that pass through a specific index are removed.
- `matrix.get_submatrix(i: int)` for an N-dimensional matrix, returns an (N-1)-dimensional matrix at some index `i`. For example, used on a 2D matrix, returns an `i`-th row.
- `matrix.as_matrix(*points: PointLike)` builds a 2D matrix out of Point-like values.

### Internal state

Matrices in `easypoint` are implemented as flat dictionaries, with empty (default) values are omitted.    
This helps with memory and speed if you have sparse matrices.

```python
matrix = Matrix((99999999, 99999999))

matrix[32474, 2387] # 0
matrix.data # {}

matrix[32474, 2387] = 327
matrix[32474, 2387] # 327
matrix.data # {3247399969913: 327}
```

If you need to swap the storage for something more efficient, build your own class.    
For example, here's an example of possible read-only `FnMatrix` class:


```python
from easypoint import Matrix
from easypoint.internal_types import Size, Index, MatrixIndexFunc


class FnMatrix(Matrix):
    def __init__(self, size: Size, fn: MatrixIndexFunc):
        self.fn = fn
        self.size = size
    
    def get_index(self, index: Index):
        return self.fn(index)
    
    def set_index(self, index: Index, value: float):
        raise ValueError("this matrix is read-only")
    
    def copy(self):
        return FnMatrix(self.size, self.fn)
    
    def new(self):
        return self.copy()
    

def sum_func(index: Index):
    x, y = index
    return x + y    


fnm = FnMatrix(sum_func)
```


# TODO

- Better docs?
- Proper conversion from list to Matrix (although it's fairly easy now)
- Better test coverage