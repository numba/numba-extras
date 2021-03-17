from typing import Generic, List, Dict, TypeVar

from jitclass import *

T = TypeVar("T")


@jitclass
class bar(Generic[T]):
    a: List[T]

    def __init__(self, v1, v2=1):
        self.a = List[T]()

        self.a.append(v1)
        self.a.append(v2)

    def append(self, value):
        self.a.append(value)

    def __getitem__(self, index):
        return self.a[index]

    def __setitem__(self, index, value):
        self.a[index] = value

    def __len__(self):
        return len(self.a)

    def get_self(self):
        return self


e = bar[int](0)


@njit
def foo():
    b = bar[int](0)

    b.append(1)

    print(len(b))
    b[0] = 100500
    print(b[0])

    d = Dict[int, str]()

    return b, d


b, d = foo()

b.append(10)
print(b.a)

print(d)

a = bar[int](-1)
a.append(3)
print(a.a)

c = bar[float](-2)
print(c.a)
print(len(c))
c[0] = 100

print(c[0])

print(c is c.get_self())


@njit
def identical(a):
    return a


print(c is identical(c))
