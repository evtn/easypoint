from easypoint import Point, FnPoint
import pytest


class TestClass:
    def test_creation(self):
        assert Point(1, 2).coords == (1, 2)
        assert Point(4, 2, 7).coords == (4, 2, 7)

        assert Point(4, 2, 7).loop == False
        assert Point(4, 2, 7, loop=True).loop == True

    def test_conversion(self):
        p1 = Point.from_(1)
        p2 = Point.from_((1, 2))
        p3 = Point.from_([3, 4, 5])
        p4 = Point.from_(p1)

        assert p1.coords == (1,)
        assert p1.loop == True

        assert p2.coords == (1, 2)
        assert p2.loop == False

        assert p3.coords == (3, 4, 5)

        assert p4 is p1

    def test_ops(self):
        p1 = Point(7, 11)
        p2 = Point(3, 5)

        assert (p1 + p2).coords == (10, 16)
        assert (p1 - p2).coords == (4, 6)
        assert (p1 * p2).coords == (21, 55)
        assert (p1 / p2).coords == (7 / 3, 11 / 5)
        assert (p1**p2).coords == (7**3, 11**5)

        assert (p1 + 8).coords == (15, 19)

        assert p1.apply(0, lambda x, _: x * 2).coords == (14, 22)

    def test_access(self):
        p1 = Point(0, 1, 2, 3, 4, 5, 6, 7)

        assert p1[6] == 6
        assert p1[2, 3].coords == (2, 3)
        assert p1[2:6].coords == (2, 3, 4, 5)
        assert p1.x == 0
        assert p1.y == 1
        assert p1.z == 2

    def test_fnpoint(self):
        fp1 = FnPoint(lambda i: i)
        fp2 = FnPoint(lambda i: i, length=3)

        assert fp1[1] == 1
        assert fp2[1] == 1
        assert fp1[48272] == 48272
        assert fp2[48272] == 0

        assert fp1[3:6].concrete().coords == (3, 4, 5, 6)
        assert fp2[3:6].concrete().coords == ()

        with pytest.raises(ValueError):
            fp1.concrete()
