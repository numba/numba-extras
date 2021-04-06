from collections import namedtuple

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

from numba import types, njit, typed
from numba.experimental import structref
from numba.experimental.structref import _Utils

from numba.core.extending import box, overload
from numba.core.imputils import lower_constant

from numba_extras.jitclass.typing_utils import (
    _GenericAlias,
    get_annotated_members,
    get_parameters,
    resolve_members,
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
    make_function,
    overload_methods,
    make_overload,
    make_constructor,
)
from numba_extras.jitclass.boxing import define_boxing
from numba_extras.jitclass.common import _Params

class ClassDescriptor:
    init: Callable
    ref_type: Type
    proxy_type: Type
    parameters: Tuple[TypeVar, ...]
    members: MembersDict
    wrapped_methods: MethodsDict
    original_methods: MethodsDict
    specificized: Dict[Tuple[Type, ...], Callable]

    def __init__(self, jitclass, cls: Type, params: _Params):
        members = params.members

        if not members:
            members = get_annotated_members(cls)

        methods = get_methods(cls)
        if "__init__" not in methods:
            methods["__init__"] = lambda self: None

        init = methods["__init__"]

        if "__new__" in methods:
            new = methods["__new__"]
            if new is Generic.__new__ or new is object.__new__:
                del methods["__new__"]

        if "__new__" in methods:
            raise NotImplementedError('Custom __new__ is not supported')

        wrapped_methods = wrap_and_jit(methods)
        parameters = get_parameters(cls)
        name = cls.__name__

        # self is not properly initialized yet. Anything except capturing may result in wierd things
        ref_meta, proxy_meta = jitclass.make_ref_and_proxy_metas(name, cls, wrapped_methods, members, params, methods, self)
        define_boxing(ref_meta, proxy_meta)

        from numba.core import cgutils
        ref_meta_inst = ref_meta({})
        @lower_constant(ref_meta_inst)
        def lower(context, builder, ty, pyval):
            import pdb; pdb.set_trace()
            obj = cgutils.create_struct_proxy(typ)(context, builder)
            return obj._getvalue()

        # init = self.init
        self.init = None
        meta_ctor = self.__make_constructor(ref_meta_inst)
        self._meta_ctor = lambda cls, *args, **kwargs: meta_ctor()
        # self.init = None

        ref_cls, proxy_cls = jitclass.make_ref_and_proxy_types(name, cls, wrapped_methods, members, params, methods, proxy_meta, self)

        define_boxing(ref_cls, proxy_cls)
        # TODO more args to 'init'

        self.init = init  # type: ignore
        self.ref_type = ref_cls
        self.proxy_type = proxy_cls
        self.parameters = parameters
        self.members = members
        self.wrapped_methods = wrapped_methods
        self.original_methods = methods
        self.specificized = {}

        if len(parameters) > 0:
            overload(proxy_cls)(make_function("init", "self", "raise NotImplementedError('Not implemented')", {}))
        else:
            ctor = self.specificize([])
            typ = ctor.__numba_class_type__
            # ovld = make_overload(methods['__call__'])
            # overload(typ)(ovld)
            from numba.core.imputils import lower_builtin

            from numba.core.typing.templates import builtin_registry

            # import pdb; pdb.set_trace()
            def get(glbls, func):
                for key, value in glbls:
                    if key == func:
                        return value

                return None
            ref_cls.__numba_call_impl__ = get(builtin_registry.globals, methods['__call__'])

            overload(proxy_cls, strict=False)(make_function("init", "*args, **kwargs", "return ctor", {'ctor': ctor.__original_func__}))

        overload_methods(methods, ref_cls)

        if '__call__' in methods:
            from numba.core.typing.templates import builtin_registry

            def get(glbls, func):
                for key, value in glbls:
                    if key == func:
                        return value

                return None

            ref_cls.__numba_call_impl__ = get(builtin_registry.globals, methods['__call__'])
            from numba.core import utils
            def _get_signature(ovld, typingctx, fnty, args, kws):
                sig = fnty.get_call_type(typingctx, args, kws)
                sig = sig.replace(pysig=utils.pysignature(ovld))
                return sig

            @lower_builtin(typ, typ, types.VarArg(types.Any))
            def method_impl(context, builder, sig, args):
                typ = sig.args[0]
                typing_context = context.typing_context
                func = typ.__numba_call_impl__
                fnty = typing_context.resolve_value_type(func)
                sig = _get_signature(func, typing_context, fnty, sig.args, {})
                call = context.get_function(fnty, sig)
                # Link dependent library
                context.add_linking_libs(getattr(call, 'libs', ()))
                return call(builder, args)

        if "__len__" in methods:
            ref_cls.__numba_len_impl__ = methods["__len__"]

        self.__add_type_mapping(cls, proxy_cls)


    def __add_type_mapping(self, cls, proxy_cls):
        def construct(args):
            _check_arguments(str(cls) + " constructor", len(self.parameters), args)
            typ = self.__specificize_type(args)
            return typ

        python_numba_type_map.add(cls, construct)
        python_numba_type_map.add(proxy_cls, construct)

        if len(self.parameters) > 0:
            @overload(cls)
            def _ovl():
                def impl():
                    # TODO raise an error
                    pass

                return impl
        else:
            ctor = self.specificize([])
            overload(cls, strict=False)(make_function("init", "*args", "return ctor", {'ctor': ctor.__original_func__}))

    def __resolve_members(self, mapped_parameters: MappedParameters) -> ResolvedMembersList:
        return resolve_members(self.members, mapped_parameters)

    def __map_parameters(self, args: Tuple[Type, ...]) -> MappedParameters:
        return {param: typ for param, typ in zip(self.parameters, args)}

    def __name_to_type_mapping(self, mapped_parameters):
        return {var.__name__: typ for var, typ in mapped_parameters.items()}

    def __make_constructor(self, struct_type: types.StructRef) -> Callable:
        # import pdb; pdb.set_trace()
        ctor = make_constructor(self.init, struct_type)
        ctor_impl = njit(ctor)
        ctor_impl.__numba_class_type__ = struct_type
        ctor_impl.__original_func__ = ctor

        return ctor_impl

    def __specificize_type(self, args: Tuple[Type, ...]) -> Tuple[NType, MappedParameters]:
        mapped_parameters = self.__map_parameters(args)
        members = self.__resolve_members(mapped_parameters)
        struct_type = self.ref_type(members)
        struct_type.mapped_parameters = self.__name_to_type_mapping(mapped_parameters)
        self.proxy_type._type.instance_type = struct_type

        # def instance_type():
        #     import pdb; pdb.set_trace()

        #     return struct_type
        # struct_type.instance_type = struct_type
        # struct_type.instance_type = lambda: None
        # import pdb; pdb.set_trace()
        # try:
        #     print(struct_type.instance_type)
        # except:
        #     pass

        return struct_type

    def __specificize_ctor(self, args: Tuple[Type, ...]) -> Callable:
        struct_type = self.__specificize_type(args)
        ctor_impl = self.__make_constructor(struct_type)

        return ctor_impl

    def specificize(self, args: List[Type]) -> Callable:
        if len(args) != len(self.parameters):
            msg = f"Wrong number of args. "
            msg += f"Expected {len(self.parameters)}({self.parameters})."
            msg += f"Got {len(args)}({args})"

            raise RuntimeError(msg)

        _args = tuple(args)
        specificized = self.specificized.get(_args)

        if not specificized:
            specificized = self.__specificize_ctor(_args)
            self.specificized[_args] = specificized

        return specificized

    def _meta_constructor(self, cls, *args, **kwargs):
        return self._meta_ctor(*args, **kwargs)
