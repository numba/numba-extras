import pytest
import typing_extensions as typing
from typing import TypeVar, List, Dict, Tuple
from numba_extras.jitclass.typing_utils import (
    get_annotated_members,
    copy_function,
    resolve_function_typevars,
    MappedParameters,
    resolve_type,
    resolve_members,
    _GenericAlias,
)
from numba import types
from numba.types.containers import ListType, DictType, Tuple as NTuple

T_int = TypeVar("T_int")
T_float = TypeVar("T_float")
T_str = TypeVar("T_str")


def test_annotated_members():
    class A:
        pass

    class B:
        a: A
        b: int
        c: float
        __private: str
        _protected: int

    expected = {"a": A, "b": int, "c": float, "_B__private": str, "_protected": int}

    actual = get_annotated_members(B)
    assert actual == expected
    assert get_annotated_members(A) == {}


def test_copy_function():
    p = 10

    def simple_function(a, b):
        return a + b

    def closure(a):
        return a + p

    def default_args(a, b=10):
        return a + b

    cases = [
        (simple_function, lambda func: func(5, 10)),
        (simple_function, lambda func: func(a=5, b=10)),
        (closure, lambda func: func(5)),
        (default_args, lambda func: func(5)),
        (default_args, lambda func: func(5, 10)),
        (default_args, lambda func: func(5, b=10)),
    ]

    for case in cases:
        func, caller = case
        cpy = copy_function(func)

        assert func is not cpy
        assert caller(func) == caller(cpy)


def test_resolve_function_typevar():
    mapped_parameters: MappedParameters = {T_int: int}

    def func():
        a = T_int

        return a

    func_res_before_resolve = func()
    resolved = resolve_function_typevars(func, mapped_parameters)

    assert func() == func_res_before_resolve
    assert resolved() == mapped_parameters[T_int]


@pytest.mark.xfail(reason="Resolving of non-global TypeVar is not supported")
def test_resolve_function_typevar_non_global():
    T_int_ = TypeVar("T_int_")
    mapped_parameters: MappedParameters = {T_int_: int}

    def func():
        a = T_int_

        return a

    func_res_before_resolve = func()
    resolved = resolve_function_typevars(func, mapped_parameters)

    assert func() == func_res_before_resolve
    assert resolved() == mapped_parameters[T_int_]


def test_resolve_type_trivial():
    cases = {
        int: types.int64,
        float: types.float64,
        str: types.unicode_type,
        T_int: types.int64,
        T_float: types.float64,
        T_str: types.unicode_type,
    }

    mapped_parameters = {T_int: int, T_float: float, T_str: str}
    for typ, expected in cases.items():
        assert resolve_type(typ, mapped_parameters) == expected


def test_resolve_type_containers():
    T_List_int = TypeVar("T_List_int")
    mapped_parameters = {T_int: int, T_float: float, T_str: str, T_List_int: List[int]}

    NList_int = ListType(types.int64)
    NListList_int = ListType(NList_int)
    NDict_str_int = DictType(types.unicode_type, types.int64)
    NDict_str_List_int = DictType(types.unicode_type, NList_int)
    NDict_int_List_int = DictType(types.int64, NList_int)
    NTuple_int_float_str = NTuple([types.int64, types.float64, types.unicode_type])
    NTuple_int_int_int = NTuple([types.int64] * 3)

    cases = {
        List[T_int]: NList_int,
        List[List[T_int]]: NListList_int,
        List[T_List_int]: NListList_int,
        Dict[T_str, T_int]: NDict_str_int,
        Dict[T_str, List[T_int]]: NDict_str_List_int,
        Dict[T_str, T_List_int]: NDict_str_List_int,
        Dict[T_int, List[T_int]]: NDict_int_List_int,
        Tuple[T_int, T_float, T_str]: NTuple_int_float_str,
        Tuple[T_int, T_int, T_int]: NTuple_int_int_int,
    }

    for typ, expected in cases.items():
        assert resolve_type(typ, mapped_parameters) == expected


def test_resolve_members():
    mapped_parameters = {T_int: int, T_float: float, T_str: str}

    members_int_float_str = {"a": T_int, "b": T_float, "c": T_str}
    members_list_int_str = {"a": List[T_int], "b": T_int, "c": T_str}
    members_int_dict = {"a": T_int, "b": Dict[T_str, T_float]}

    expected_members_int_float_str = [
        ("a", types.int64),
        ("b", types.float64),
        ("c", types.unicode_type),
    ]
    expected_members_list_int_str = [
        ("a", ListType(types.int64)),
        ("b", types.int64),
        ("c", types.unicode_type),
    ]
    expected_members_int_dict = [
        ("a", types.int64),
        ("b", DictType(types.unicode_type, types.float64)),
    ]

    cases = [
        (members_int_float_str, expected_members_int_float_str),
        (members_list_int_str, expected_members_list_int_str),
        (members_int_dict, expected_members_int_dict),
    ]

    for members, expected in cases:
        assert resolve_members(members, mapped_parameters) == expected


def test_generic_alias():
    class A:
        pass

    class B:
        pass

    assert _GenericAlias(A, int) is _GenericAlias(A, int)
    assert _GenericAlias(A, int) is not _GenericAlias(A, float)
    assert _GenericAlias(B, int) is not _GenericAlias(A, int)
