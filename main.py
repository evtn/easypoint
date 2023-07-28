from easypoint.point import FnPoint, Point
from easypoint.matrix import Matrix, _index_to_i


# matrix = Matrix((10, 5, 4))

# for index in matrix:
#     matrix[index] = _index_to_i(index, matrix.size)


# m = Point.as_matrix((1, 0), (0, 1))
# print(m.to_list())
# print(Point(4, 5).transform(m))

p1 = FnPoint(lambda x: x**2)

p2 = p1[10:]

p3 = p2[:10]

p4 = p3.concrete()

print(p1[:20].concrete().named("A"))
print(p2[:20].concrete().named("B"))
print(p3[:20].concrete().named("C"))
print(p4[:20].named("D"))
