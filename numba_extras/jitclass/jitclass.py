from collections import namedtuple
import sys
import operator

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
import types as ptypes

from numba import types, njit, typed
from numba.experimental import structref
from numba.experimental.structref import _Utils
from numba.core.serialize import ReduceMixin

from numba.core.extending import box, unbox, NativeValue, overload

import weakref
import numba_extras.jitclass.rewrites.call_rewrite

from numba_extras.jitclass.typing_utils import (
    _GenericAlias,
    get_annotated_members,
    get_parameters,
    resolve_type,
    resolve_members,
    get_class,
    MappedParameters,
    MembersDict,
    MethodsDict,
    ResolvedMembersList,
    NType,
)
from numba_extras.jitclass.typemap import python_numba_type_map, _check_arguments
from numba_extras.jitclass.overload_utils import (
    get_methods,
    wrap_and_jit,
    make_python_wrapper,
    make_property,
    make_function,
    overload_methods,
    make_constructor,
)
from numba_extras.jitclass.serialization_utils import (
    class_info_attr,
    ClassInfo,
    StructRefProxyMeta,
    SerializableStructRefProxyMetaType,
    SerializableBoxing,
    SerializableStructRef,
    StructRefProxy,
    StructRefProxyMetaType,
    GenericStructRefProxyMetaType,
)
from numba_extras.jitclass.class_descriptor import ClassDescriptor
from numba_extras.jitclass.common import _Params

_Params = namedtuple("_Params", ["compile_methods", "jit_options", "members"])


class method_option:
    OptionsType = Dict[str, Any]

    __attr = "__jitclass_method_options"

    overload_attr = "overload"
    compile_attr = "compile"

    all_options = [overload_attr, compile_attr]

    def __init__(self, func):
        self._func = func

    def __getattr__(self, name):
        if name in self.all_options:
            options = self.get_options(self._func)
            return options.get(name)

        raise AttributeError(f"'{self}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name in self.all_options:
            self.update_options(self._func, {name: value})
        elif name == "_func":
            super().__setattr__(name, value)
        else:
            raise AttributeError(f"'{self}' object has no attribute '{name}'")

    @classmethod
    def get_options(cls, func: Callable) -> OptionsType:
        try:
            return getattr(func, cls.__attr)
        except AttributeError:
            return {}

    @classmethod
    def update_options(cls, func: Callable, options: OptionsType) -> None:
        _options = cls.get_options(func)
        _options.update(options)
        setattr(func, cls.__attr, _options)


class jitclass:
    # type aliases
    ClassTypes = Dict[Type, ClassDescriptor]
    JitClassCache = Dict[Type, StructRefProxy]
    Members = Optional[Dict[str, type]]
    Options = Optional[Dict[str, Any]]

    MembersDict = Dict[str, Any]
    MethodsDict = Dict[str, Callable]

    # class variables
    __class_types: ClassVar[ClassTypes] = {}
    __cache: ClassVar[JitClassCache] = {}
    __meta_cache: ClassVar[JitClassCache] = {}

    # members
    params: _Params

    def __new__(
        cls: Type,
        *args: Type,
        compile_methods: bool = True,
        jit_options: Options = None,
        members: Members = None,
    ):
        if len(args):
            if len(args) > 1:
                raise RuntimeError("Something went wrong")

            origin_class = args[0]
            if not isinstance(origin_class, type):
                raise RuntimeError("Something went wrong")

            return jitclass(
                compile_methods=compile_methods,
                jit_options=jit_options,
                members=members,
            )(origin_class)

        return super().__new__(cls)

    def __init__(
        self: Any,
        *args: Type,
        compile_methods: bool = True,
        jit_options: Options = None,
        members: Members = None,
    ):
        if not isinstance(self, jitclass):
            return None

        if len(args):
            raise RuntimeError("Something went wrong")

        self.params = _Params(compile_methods, jitclass, members)

    @staticmethod
    def __make_proxy_class_meta__(cls: Type, params: _Params):
        proxy_meta = StructRefProxyMetaType(cls, params)

        @overload(proxy_meta)
        def meta_ovld():
            def impl():
                return None

            return impl

        return proxy_meta

    def __call__(self, cls: Type) -> Type:
        if cls in self.__cache:
            return self.__cache[cls]

        if not issubclass(cls, typing.Generic):
            meta = self.__make_proxy_class_meta__(cls, self.params)
            jitclass.__meta_cache[cls] = meta
            proxy_class = meta()
            self.__cache[cls] = proxy_class

            return proxy_class
        else:
            meta = GenericStructRefProxyMetaType(cls, self.params)
            jitclass.__meta_cache[cls] = meta
            proxy_class = meta()
            self.__cache[cls] = proxy_class

            return proxy_class

        return proxy_type

    @staticmethod
    def __list_getitem(_cls, args):
        typ = resolve_type(args[0], {})

        @njit
        def ctor_impl():
            return typed.List.empty_list(typ)

        ctor_impl.__numba_class_type__ = python_numba_type_map.construct(_cls, [typ])

        def ovl_impl(cls, args):
            return ctor_impl

        return ovl_impl

    @staticmethod
    def __dict_getitem(_cls, args):
        key_type = resolve_type(args[0], {})
        value_type = resolve_type(args[1], {})

        @njit
        def ctor_impl():
            return typed.Dict.empty(key_type=key_type, value_type=value_type)

        ctor_impl.__numba_class_type__ = python_numba_type_map.construct(
            _cls, [key_type, value_type]
        )

        def ovl_impl(cls, args):
            return ctor_impl

        return ovl_impl

    @staticmethod
    @overload(operator.getitem)
    def __class_getitem(cls, args):
        _cls = get_class(cls)
        if _cls is None:
            return None

        if not isinstance(args, (types.containers.Tuple, types.containers.UniTuple)):
            args = (args,)

        args = [get_class(arg) for arg in args]
        if not all(args):
            # TODO raise an error
            return None

        if _cls is List:
            return jitclass.__list_getitem(_cls, args)

        if _cls is Dict:
            return jitclass.__dict_getitem(_cls, args)

        class_descr = jitclass.__class_types.get(_cls)
        if class_descr is None:
            # TODO raise an error
            return None

        ctor_impl = class_descr.specificize(args)

        def ovl_impl(cls, args):
            return ctor_impl

        return ovl_impl

    @staticmethod
    def _get_trivial_special_methods(class_descr):
        def __new__(self, *args, **kwargs):
            specificized = class_descr.specificize([])
            return specificized(*args, **kwargs)

        return {
            "__new__": __new__,
        }

    @staticmethod
    def _get_generic_special_methods(class_descr):
        def __init__(self, *args, **kwargs):  # make valid args
            raise NotImplementedError("Not implemented")

        @classmethod
        def __class_getitem__(cls, params):
            alias = _GenericAlias(cls, params)

            return alias

        @classmethod
        def initializer(cls, params, *args, **kwargs):
            specificized = class_descr.specificize(params)

            return specificized

        # TODO figure out if we need it?
        def __new__(cls, *args, **kwargs):
            return type.__new__(cls, *args, **kwargs)

        return {
            "__new__": __new__,
            "__init__": __init__,
            "__class_getitem__": __class_getitem__,
            "__initialize__": initializer,
        }

    @staticmethod
    def get_special_methods(class_descr, template_parameters):
        if len(template_parameters) > 0:
            special_methods = jitclass._get_generic_special_methods(class_descr)
        else:
            special_methods = jitclass._get_trivial_special_methods(class_descr)

        return special_methods

    @staticmethod
    def wraped_members(name: str, methods: MethodsDict, members: MembersDict):
        # TODO private methods
        all_members = {
            method_name: make_python_wrapper(method)
            for method_name, method in methods.items()
        }

        # TODO private members
        members = {
            member_name: make_property(name, member_name)
            for member_name in members.keys()
        }
        all_members.update(members)

        return all_members

    @staticmethod
    def make_class_info(cls: Type, parameters: _Params):
        return {class_info_attr: ClassInfo(cls, parameters)}

    @staticmethod
    def __make_proxy_class_meta(
        name: str,
        cls: Type,
        methods: MethodsDict,
        members: MembersDict,
        params: _Params,
        class_descr: ClassDescriptor,
    ):
        class_info = jitclass.make_class_info(cls, params)

        def ctor(cls, name, bases, dct):
            return class_descr._meta_constructor(cls, name, bases, dct)

        proxy_meta = SerializableStructRefProxyMetaType(
            name, {**class_info, "ctor": ctor}
        )

        jitclass.__meta_cache[cls] = proxy_meta

        @overload(proxy_meta)
        def meta_ovld():
            def impl():
                return None

            return impl

        return proxy_meta

    # creates python proxy for our jitclass
    @staticmethod
    def __make_proxy_class(
        name: str,
        cls: Type,
        methods: MethodsDict,
        members: MembersDict,
        params: _Params,
        meta: Type,
        class_descr: ClassDescriptor,
    ):
        wrapped_members = jitclass.wraped_members(name, methods, members)
        template_parameters = get_parameters(cls)
        special_methods = jitclass.get_special_methods(class_descr, template_parameters)
        class_info = jitclass.make_class_info(cls, params)

        members = {**wrapped_members, **special_methods, **class_info}
        proxy_class = meta.create(members)

        return proxy_class

    @staticmethod
    def __make_ref_class(name, proxy_cls):
        return SerializableStructRef.create(name, proxy_cls, None)

    @staticmethod
    def make_ref_and_proxy_types(
        name: str,
        cls: Type,
        wrapped_methods: MethodsDict,
        members: MembersDict,
        params: _Params,
        methods: MethodsDict,
        meta: Type,
        class_descr: ClassDescriptor,
    ):
        proxy_cls = jitclass.__make_proxy_class(
            name, cls, wrapped_methods, members, params, meta, class_descr
        )
        ref_cls = jitclass.__make_ref_class(f"{name}_ref", proxy_cls)

        jitclass.__cache[cls] = proxy_cls

        return ref_cls, proxy_cls

    @staticmethod
    def make_ref_and_proxy_metas(
        name: str,
        cls: Type,
        wrapped_methods: MethodsDict,
        members: MembersDict,
        params: _Params,
        methods: MethodsDict,
        class_descr: ClassDescriptor,
    ):
        proxy_meta = jitclass.__make_proxy_class_meta(
            name, cls, wrapped_methods, members, params, class_descr
        )
        ref_meta = jitclass.__make_ref_class(f"{proxy_meta.__name__}_ref", proxy_meta)

        return ref_meta, proxy_meta

    @classmethod
    def method_options(cls, func: Callable, values: Dict[str, Any]):
        method_option.update_options(func, values)
        return func

    @classmethod
    def overload_method(cls, func: Callable):
        compile = method_option.compile_attr
        overload = method_option.overload_attr
        return cls.method_options(func, {compile: True, overload: True})

    @classmethod
    def compile(cls, func):
        compile = method_option.compile_attr
        return cls.method_options(func, {compile: True})

    @classmethod
    def do_not_compile(cls, func):
        compile = method_option.compile_attr
        return cls.method_options(func, {compile: False})

    @classmethod
    def get_meta(cls, orig_class, params):
        return cls.__meta_cache[orig_class]
