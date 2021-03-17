import typing_extensions as typing
from typing import (
    Dict,
    Type,
    Optional,
    Any,
    Callable,
    ClassVar,
    Generic,
    List,
    Mapping,
    Tuple,
    TypeVar,
)

from numba_extras.jitclass.jitclass import jitclass

class jitclass:
    Members = Optional[Dict[str, type]]
    Options = Optional[Dict[str, Any]]

    MembersDict = Dict[str, Any]
    MethodsDict = Dict[str, Callable]

    def __new__(
        cls: Type, *args: Type, compile_methods: bool = True, jit_options: Options = None, members: Members = None,
    ):
        return jitclass.__new__(jitclass, *args, compile_methods=compile_methods, jit_options=jit_options, members=members)

    @staticmethod
    def method_options(func: Callable, values: Dict[str, Any]):
        return jitclass.method_options(func, values)

    @staticmethod
    def overload_method(cls, func: Callable):
        return jitclass.overload_method(func)

    @staticmethod
    def compile(cls, func: Callable):
        return jitclass.compile(func)

    @classmethod
    def do_not_compile(cls, func: Callable):
        return jitclass.do_not_compile(func)
