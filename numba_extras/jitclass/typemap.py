import typing_extensions as typing
from typing import TypeVar, List, Dict, Type, Tuple, Callable, Union, _SpecialForm

import numba
from numba.core import types
from numba.core.extending import overload
from numba.core.types.functions import Function
from numba.core.types import Type as NType

PType = Union[Type, _SpecialForm]


class _PythonNumbaTypeMap:
    Constructor = Callable[[List[NType]], NType]
    map: Dict[PType, Constructor]

    def __init__(self):
        self.map = {}

    def add(self, typ: PType, constructor: Constructor) -> None:
        if typ in self.map:
            raise RuntimeError(
                f"Can't add new constructor for type {typ}. Constructor alredy exists"
            )

        self.map[typ] = constructor

    def construct(self, typ: PType, args: List[NType]) -> NType:
        if typ not in self.map:
            raise RuntimeError(
                f"Can't construct Numba equivalent for type {typ}. No known constructors"
            )

        return self.map[typ](args)

    def __contains__(self, typ):
        return typ in self.map


def _check_arguments(func_name: str, expected: int, args: List[NType]):
    actual = len(args)
    if actual != expected:
        raise RuntimeError(
            f"{func_name} takes exactly {expected} arguments. {actual} provided: {args}"
        )


def _check_type(typs):
    for typ in typs:
        # import pdb; pdb.set_trace()
        if not isinstance(typ, NType):
            raise RuntimeError(
                f"Can't construct Numba type with non-Numba type parameter {typ}"
            )


def _int_constructor(args: List[NType]) -> NType:
    _check_arguments("_int_constructor", 0, args)
    return types.int64


def _float_constructor(args: List[NType]) -> NType:
    _check_arguments("_float_constructor", 0, args)
    return types.float64


def _str_constructor(args: List[NType]) -> NType:
    _check_arguments("_str_constructor", 0, args)
    return types.unicode_type


def _list_constructor(args: List[NType]) -> NType:
    _check_arguments("_list_constructor", 1, args)
    _check_type(args)
    return types.containers.ListType(args[0])


def _dict_constructor(args: List[NType]) -> NType:
    _check_arguments("_dict_constructor", 2, args)
    _check_type(args)
    return types.containers.DictType(args[0], args[1])


def _tuple_constructor(args: List[NType]) -> NType:
    if len(args) == 0:
        raise RuntimeError(
            "At least one argument should be provided to _tuple_constructor. 0 provided"
        )

    _check_type(args)
    return types.containers.Tuple(args)


python_numba_type_map = _PythonNumbaTypeMap()

python_numba_type_map.add(int, _int_constructor)
python_numba_type_map.add(float, _float_constructor)
python_numba_type_map.add(str, _str_constructor)
python_numba_type_map.add(list, _list_constructor)
python_numba_type_map.add(List, _list_constructor)
python_numba_type_map.add(dict, _dict_constructor)
python_numba_type_map.add(Dict, _dict_constructor)
python_numba_type_map.add(tuple, _tuple_constructor)
python_numba_type_map.add(Tuple, _tuple_constructor)


@overload(List)
def list_ovl():
    def impl():
        # TODO raise an error
        pass

    return impl


@overload(Dict)
def dict_ovl():
    def impl():
        # TODO raise an error
        pass

    return impl


@overload(Tuple)
def tuple_ovl():
    def impl():
        # TODO raise an error
        pass

    return impl
