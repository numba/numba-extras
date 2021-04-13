import re
import typing_extensions as typing
from typing import List, Dict, Tuple

from numba_extras.jitclass import typemap
from numba.types import int8, int64, float64, unicode_type
from numba.types.containers import ListType, DictType, Tuple as NTuple
from contextlib import contextmanager, AbstractContextManager

import itertools
from tutils import raises_with_msg


params_count_mismatch_msg = re.compile(
    r".+ takes exactly [0-9]+ arguments\. [0-9]+ provided: \[.*\]"
)
params_type_mismatch_msg = re.compile(
    r"Can't construct Numba type with non-Numba type parameter .+"
)

trivial_python_types = [float, int, str]
trivial_numba_types = [float64, int64, unicode_type]
list_types = [list, List]
dict_types = [dict, Dict]
tuple_types = [tuple, Tuple]


def test_map_add():
    class _int8:
        pass

    typemap.python_numba_type_map.add(_int8, lambda args: int8)
    assert _int8 in typemap.python_numba_type_map


def test_map_add_already_existed():
    msg = re.compile(
        r"Can't add new constructor for type .+\. Constructor alredy exists"
    )

    with raises_with_msg(RuntimeError, msg):
        typemap.python_numba_type_map.add(int, lambda args: int64)


def test_map_construct_unknown():
    class _int8:
        pass

    msg = re.compile(
        r"Can't construct Numba equivalent for type .+\. No known constructors"
    )

    with raises_with_msg(RuntimeError, msg):
        constructed = typemap.python_numba_type_map.construct(_int8, [])


def test_trivial_types():
    for ptype, ntype in zip(trivial_python_types, trivial_numba_types):
        constructed = typemap.python_numba_type_map.construct(ptype, [])
        assert constructed == ntype


def test_trivial_types_wrong_args():
    for ptype in trivial_python_types:
        with raises_with_msg(RuntimeError, params_count_mismatch_msg):
            typemap.python_numba_type_map.construct(ptype, [int64])


def test_list():
    numba_types = [ListType(typ) for typ in trivial_numba_types]

    for itype, ntype in zip(trivial_numba_types, numba_types):
        for ltype in list_types:
            constructed = typemap.python_numba_type_map.construct(ltype, [itype])
            constructed == ntype


def test_list_wrong_args_count():
    for ltype in list_types:
        with raises_with_msg(RuntimeError, params_count_mismatch_msg):
            constructed = typemap.python_numba_type_map.construct(ltype, [])


def test_list_wrong_args_type():
    for ltype in list_types:
        with raises_with_msg(RuntimeError, params_type_mismatch_msg):
            constructed = typemap.python_numba_type_map.construct(ltype, [int])


def test_dict():
    inner_types = itertools.product(trivial_numba_types, trivial_numba_types)
    inner_types = [(key, value) for key, value in inner_types]
    numba_types = [DictType(key, value) for key, value in inner_types]

    for itype, ntype in zip(inner_types, numba_types):
        for dtype in dict_types:
            constructed = typemap.python_numba_type_map.construct(dtype, list(itype))
            constructed == ntype


def test_dict_wrong_args_count():
    for dtype in dict_types:
        with raises_with_msg(RuntimeError, params_count_mismatch_msg):
            constructed = typemap.python_numba_type_map.construct(dtype, [])


def test_dict_wrong_args_type():
    for dtype in dict_types:
        with raises_with_msg(RuntimeError, params_type_mismatch_msg):
            constructed = typemap.python_numba_type_map.construct(dtype, [int, int])


def test_tuple():
    inner_types = itertools.product(trivial_numba_types, trivial_numba_types)
    inner_types = [(t1, t2) for t1, t2 in inner_types]
    numba_types = [NTuple([t1, t2]) for t1, t2 in inner_types]

    for itype, ntype in zip(inner_types, numba_types):
        for ttype in tuple_types:
            constructed = typemap.python_numba_type_map.construct(ttype, list(itype))
            constructed == ntype


def test_tuple_wrong_args_count():
    msg = "At least one argument should be provided to _tuple_constructor. 0 provided"

    for ttype in tuple_types:
        with raises_with_msg(RuntimeError, msg):
            constructed = typemap.python_numba_type_map.construct(ttype, [])


def test_tuple_wrong_args_type():
    for ttype in tuple_types:
        with raises_with_msg(RuntimeError, params_type_mismatch_msg):
            constructed = typemap.python_numba_type_map.construct(ttype, [int, int])
