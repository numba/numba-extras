import re
import pytest
import typing_extensions as typing
from typing import Generic, TypeVar, List, Dict, Tuple
from numba_extras.jitclass.typing_utils import (
    get_annotated_members,
    copy_function,
    resolve_function_typevars,
    MappedParameters,
    resolve_type,
    resolve_members,
)
from numba_extras.jitclass.jitclass import jitclass
from numba import types
from numba import njit
from numba.types.containers import ListType, DictType, Tuple as NTuple
from numba.typed import List as TList
from numba.core.errors import TypingError

from tutils import raises_with_msg

T = TypeVar("T")

T0 = TypeVar("T0")
T1 = TypeVar("T1")
T2 = TypeVar("T1")

class TrivialSingleMember:
    a: int

    def __init__(self, a):
        self.a = a

    def __add__(self, rhs):
        return TrivialSingleMember(self.a + rhs.a)

    def __sub__(self, rhs):
        return TrivialSingleMember(self.a - rhs.a)

    def __mul__(self, rhs):
        return TrivialSingleMember(self.a * rhs.a)

    def __floordiv__(self, rhs):
        return TrivialSingleMember(self.a // rhs.a)

    def __eq__(self, rhs):
        return self.a == rhs.a

    def __ne__(self, rhs):
        return self.a != rhs.a

    def __call__(self, a):
        return self.a + a


class SingleMember(Generic[T]):
    a: T

    def __init__(self, a):
        self.a = a

    def __add__(self, rhs):
        return SingleMember[T](self.a + rhs.a)

    def __sub__(self, rhs):
        return SingleMember[T](self.a - rhs.a)

    def __mul__(self, rhs):
        return SingleMember[T](self.a * rhs.a)

    def __floordiv__(self, rhs):
        return SingleMember[T](self.a // rhs.a)

    def __eq__(self, rhs):
        return self.a == rhs.a

    def __ne__(self, rhs):
        return self.a != rhs.a


class MultipleMembers(Generic[T0, T1, T2]):
    a: T0
    b: T1
    c: T2

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c


class SingleList(Generic[T]):
    a: List[T]

    def __init__(self, a):
        self.a = a

    def __getitem__(self, index):
        return self.a[index]

    def __setitem__(self, index, value):
        self.a[index] = value

    def __len__(self):
        return len(self.a)

    def append(self, value):
        self.a.append(value)


class NestedSingleMember(Generic[T]):
    a: SingleMember[T]

    def __init__(self, a):
        self.a = SingleMember[T](a)


class SingleMemberOverload(Generic[T]):
    a: T

    def __init__(self, a):
        self.a = a

    def plus_2(self):
        return self.a + 2

    def minus_2(self):
        return self.a - 2

    @jitclass.overload_method
    def plus_minus_2(self):
        if self.mapped_parameters["T"] == int:
            return SingleMemberOverload.plus_2

        if self.mapped_parameters["T"] == float:
            return SingleMemberOverload.minus_2

        raise NotImplementedError()


def test_jitclass_is_same():
    jitted1 = jitclass(SingleMember)
    jitted2 = jitclass(SingleMember)

    assert jitted1 == jitted2


def test_single_member_python():
    jitted = jitclass(SingleMember)
    interpreted = SingleMember
    # import pdb; pdb.set_trace()

    def get_list():
        l = TList.empty_list(types.int64)
        l.append(10)

        return l

    cases = {
        int: lambda x: x(10),
        float: lambda x: x(10.0),
        str: lambda x: x("10"),
        List[int]: lambda x: x(get_list()),
    }

    for typ, init in cases.items():
        j_instance = init(jitted[typ])
        i_instance = init(interpreted[typ])

        assert j_instance.a == i_instance.a


def test_single_member_jit():
    jitted = jitclass(SingleMember)
    interpreted = SingleMember

    @njit
    def get_list():
        ls = TList.empty_list(types.int64)
        ls.append(10)

        return ls

    cases = [
        lambda: SingleMember[int](10),
        lambda: SingleMember[float](10.0),
        lambda: SingleMember[str]("10"),
        lambda: SingleMember[List[int]](get_list()),
    ]

    for init in cases:
        j_instance = njit(init)()
        i_instance = init()

        assert j_instance.a == i_instance.a


def test_single_member_operators_python_jit():
    jitted = jitclass(SingleMember)
    interpreted = SingleMember

    def run(cases, operators):
        for typ, init in cases.items():
            j_instance = init(jitted[typ])
            i_instance = init(interpreted[typ])
            for name, op in operators.items():
                j_result = op(j_instance)
                nj_result = njit(op)(j_instance)
                i_result = op(i_instance)

                assert i_result == j_result, f"operator {name} failed for type {interpreted[typ]}"
                assert i_result == nj_result, f"operator {name} failed for type {interpreted[typ]}"

    numeric_cases = {int: lambda x: x(10), float: lambda x: x(10.0)}

    string_cases = {str: lambda x: x("10")}

    numeric_operators = {
        "add": lambda x: x + x,
        "sub": lambda x: x - x,
        "mul": lambda x: x * x,
        "floordiv": lambda x: x // x,
    }

    string_operators = {"add": lambda x: x + x}

    logic_operators = {"eq": lambda x: x == x, "ne": lambda x: x != x}

    run(numeric_cases, numeric_operators)
    run(numeric_cases, logic_operators)
    run(string_cases, string_operators)
    run(string_cases, logic_operators)


def test_single_member_unboxing():
    jitted = jitclass(SingleMember)
    interpreted = SingleMember

    cases = {int: lambda x: x(10), float: lambda x: x(10.0), str: lambda x: x("10")}

    for typ, init in cases.items():

        @njit
        def jinit(v):
            return jitted[typ](v)

        j_instance = init(jinit)
        i_instance = init(interpreted[typ])

        assert j_instance.a == i_instance.a


def test_single_member_boxing():
    jitted = jitclass(SingleMember)
    interpreted = SingleMember

    cases = {int: lambda x: x(10), float: lambda x: x(10.0), str: lambda x: x("10")}

    for typ, init in cases.items():

        @njit
        def get_a(instance):
            return instance.a

        j_value = get_a(init(jitted[typ]))
        i_instance = init(interpreted[typ])

        assert j_value == i_instance.a


def test_multiple_members_python():
    js = jitclass(SingleMember)
    jitted = jitclass(MultipleMembers)
    interpreted = MultipleMembers

    cases = [
        ((int, float, str), lambda x: x(10, 10.0, "10")),
        ((js[int], js[float], js[str]), lambda x: x(js[int](10), js[float](10.0), js[str]("10")),),
    ]

    for types, init in cases:
        ji = init(jitted[types])
        ii = init(interpreted[types])
        assert (ji.a, ji.b, ji.c) == (ii.a, ii.b, ii.c)


def test_nested_single_method_python():
    js = jitclass(SingleMember)
    jitted = jitclass(NestedSingleMember)
    interpreted = NestedSingleMember

    cases = {int: lambda x: x(10), float: lambda x: x(10.0), str: lambda x: x("10")}

    for typ, init in cases.items():
        ji = init(jitted[typ])
        ii = init(interpreted[typ])

        assert ii.a == ji.a


def test_nested_single_method_jit():
    js = jitclass(SingleMember)
    jitted = jitclass(NestedSingleMember)
    interpreted = NestedSingleMember

    cases = [
        lambda: NestedSingleMember[int](10),
        lambda: NestedSingleMember[float](10.0),
        lambda: NestedSingleMember[str]("10"),
    ]

    for init in cases:
        ji = njit(init)()
        ii = init()

        assert ii.a == ji.a


def test_single_list():
    # js = jitclass(SingleMember)
    jitted = jitclass(SingleList)
    interpreted = SingleList

    def get_list(typ, val):
        ls = TList.empty_list(typ)
        ls.append(val)

        return ls

    cases = {
        int: lambda x: x(get_list(types.int64, 10)),
        float: lambda x: x(get_list(types.float64, 10.0)),
        str: lambda x: x(get_list(types.unicode_type, "10")),
    }
    #  js[int]: lambda x: x(get_list(js[int], js[int](10)))}

    for typ, init in cases.items():
        j_instance = init(jitted[typ])
        i_instance = init(interpreted[typ])

        assert j_instance.a == i_instance.a
        assert j_instance[0] == i_instance[0]

        j_instance.append(j_instance[0])
        i_instance.append(i_instance[0])

        assert len(j_instance) == len(i_instance)
        assert j_instance[1] == i_instance[1]

        j_instance[1] = 2 * j_instance[1]
        i_instance[1] = 2 * i_instance[1]

        assert j_instance[1] == i_instance[1]


# @pytest.mark.xfail(reason="Creating TypeList of jitclasses is not supported yet")
# def test_single_list_single_member():
#     js = jitclass(SingleMember)
#     jitted = jitclass(SingleList)
#     interpreted = SingleList

#     def get_list(typ, val):
#         import pdb; pdb.set_trace()
#         ls = TList.empty_list(typ)
#         # import pdb; pdb.set_trace()
#         ls.append(val)

#         return ls

#     cases = {js[int]: lambda x: x(get_list(js[int], js[int](10)))}

#     for typ, init in cases.items():
#         j_instance = init(jitted[typ])
#         i_instance = init(interpreted[typ])

#         assert j_instance.a == i_instance.a
#         assert j_instance[0] == i_instance[0]

#         j_instance.append(j_instance[0])
#         i_instance.append(i_instance[0])

#         assert len(j_instance) == len(i_instance)

#         j_instance[1] = 2 * j_instance[1]
#         i_instance[1] = 2 * i_instance[1]

#         assert j_instance[1] == i_instance[1]


def test_single_member_overload():
    jitted = jitclass(SingleMemberOverload)
    interpreted = SingleMemberOverload

    ji_int = jitted[int](0)
    ii_int = interpreted[int](0)

    ji_float = jitted[float](0)
    ii_float = interpreted[float](0)

    ji_str = jitted[str]("0")

    assert ji_int.plus_minus_2() == ji_int.plus_2()
    assert ji_float.plus_minus_2() == ji_float.minus_2()

    msg = re.compile(r".*NotImplementedError.*", re.MULTILINE | re.DOTALL)
    with raises_with_msg(TypingError, msg):
        ji_str.plus_minus_2()

def test_trivial_single_member():
    jitted = jitclass(TrivialSingleMember)
    interpreted = TrivialSingleMember

    ji = jitted(10)
    ii = interpreted(10)

    assert ii == ji

def test_trivial_single_member_jit():
    jitted = jitclass(TrivialSingleMember)
    interpreted = TrivialSingleMember

    # import pdb; pdb.set_trace()
    def init():
        return TrivialSingleMember(10)

    # import pdb; pdb.set_trace()
    # ji = njit(init)()
    ji = jitted(10)
    ii = init()

    assert ii == ji

# call = TrivialSingleMember.__call__

def test_callable():
    jitted = jitclass(TrivialSingleMember)

    @njit
    def test():
        a = jitted(10)

        return a(32)

    test()

def test_generic():
    jitted = jitclass(SingleMember)

    jitted[int]
# def test_pass_type():
#     jitted = jitclass(TrivialSingleMember)
#     # tj = type(jitted)

#     import pdb; pdb.set_trace()
#     # @njit
#     # def test():
#     #     return tj()
#     @njit
#     def test(typ):
#         return typ(10)
#         # a = typ(10)

#         # return a(32)

#     a = test(jitted)
#     # test()
#     # a = jitted(10)
#     # import pdb; pdb.set_trace()
#     print(a)

# def test_single_list_trivial():
#     interpreted = TrivialSingleMember
#     jitted = jitclass(TrivialSingleMember)

#     @njit
#     def get_list(typ, val):
#         ls = TList.empty_list(typ)
#         ls.append(val)

#         return ls

#     # jitted(10)
#     # ls = get_list(jitted, jitted(10))
#     ls = get_list(types.int64, 10)

import sys

# test_trivial_single_member_jit()
test_generic()
# test_pass_type()
# test_single_list_trivial()