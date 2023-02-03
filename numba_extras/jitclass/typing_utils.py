import types as ptypes
import typing_extensions as typing
from typing import (
    get_type_hints,
    TypeVar,
    Type,
    Tuple,
    List,
    Dict,
    Union,
    Any,
    Callable,
    Optional,
    Mapping,
    ClassVar,
)
from typing import _GenericAlias as TGenericAlias  # type: ignore
from types import FunctionType
import functools
import copy
import weakref

from numba_extras.jitclass.typemap import python_numba_type_map, PType

# from numba_extras.jitclass.serialization_utils import SerializableStructRefProxyMeta
# from numba_extras.jitclass.serialization_utils import StructRefProxySerializer
import numba
from numba.core.types.functions import Function
from numba.core.types import Type as NType

from numba.core.serialize import ReduceMixin

TypeOrGeneric = Union[Type, "_GenericAlias"]
TypeOrTuple = Union[TypeOrGeneric, Tuple[TypeOrGeneric, ...]]
MappedParameters = Dict[TypeVar, TypeOrGeneric]
MemberType = Union[TypeOrGeneric, TypeVar, str]
MembersDict = Dict[str, MemberType]
MethodsDict = Dict[str, Callable]
ResolvedMembersList = List[Tuple[str, NType]]

import abc

# need to override behavior of typing._GenericAlias to pass
# templates argument into constructor in the following case:
# klass[int](...)
class _GenericAlias(ReduceMixin):
    CacheType = Mapping[Tuple[TypeOrGeneric, ...], "_GenericAlias"]
    __cache: ClassVar[CacheType] = weakref.WeakValueDictionary()

    def __new__(cls, wrapee: Type, params: TypeOrTuple):
        try:
            return cls.__cache[cls.__make_key(wrapee, params)]
        except (KeyError, TypeError):
            return super().__new__(cls)

    def __init__(self, wrapee: Type, params: TypeOrTuple):
        if hasattr(self, "_GenericAlias__alias"):
            return

        self.__alias = TGenericAlias(wrapee, params)
        try:
            self.__cache[self.__make_key(wrapee, params)] = self  # type: ignore
        except TypeError:
            pass

    def __getitem__(self, params: TypeOrTuple):
        return self.__alias.__getitem__(params)

    def __repr__(self):
        return self.__alias.__repr__()

    def __eq__(self, other):
        return self.__alias.__eq__(other)

    def __hash__(self):
        return self.__alias.__hash__()

    def __call__(self, *args, **kwargs):
        # import pdb; pdb.set_trace()
        return self.__alias.__origin__.__initialize__(self.__alias.__args__)(
            *args, **kwargs
        )

    def __mro_entries__(self, bases):
        return self.__alias.__mro_entries__(bases)

    def __getattr__(self, name: str):
        if name == "_numba_type_":
            try:
                specificized = self.__alias.__origin__.__initialize__(
                    self.__alias.__args__
                )
                return getattr(specificized, "__numba_class_type__")
            except:
                pass

        if name == "_GenericAlias__alias":
            raise AttributeError()

        return getattr(self.__alias, name)

    def __setattr__(self, attr: str, val: Any):
        if attr != "_GenericAlias__alias":
            return setattr(self.__alias, attr, val)
        else:
            return super().__setattr__(attr, val)

    def __instancecheck__(self, obj):
        return self.__origin__.__instancecheck__(obj)

    # def __instancecheck__(self, obj):
    #     return self.__subclasscheck__(type(obj))

    # def __subclasscheck__(self, cls, other = None):
    #     import pdb; pdb.set_trace()
    #     return self.__alias.__subclasscheck__(cls)

    # def __reduce__(self):
    #     return self.__alias.__reduce__()

    @classmethod
    def __make_key(cls, wrapee: Type, params: TypeOrTuple):
        if not isinstance(params, tuple):
            params = (params,)

        return (wrapee,) + params

    def _reduce_states(self):
        return {"wrapee": self.__origin__._reduce_states(), "params": self.__args__}

    @classmethod
    def _rebuild(cls, wrapee, params):
        from numba_extras.jitclass.serialization_utils import StructRefProxySerializer

        # wrapee = SerializableStructRefProxyMeta._rebuild(**wrapee)
        wrapee = StructRefProxySerializer._rebuild(**wrapee)

        return cls(wrapee, params)


def copy_function(func: FunctionType) -> FunctionType:
    # TODO: copy __code__ object
    cpy = FunctionType(
        func.__code__,
        copy.copy(func.__globals__),
        name=func.__name__,
        argdefs=copy.copy(func.__defaults__),
        closure=copy.copy(func.__closure__),
    )
    cpy.__dict__ = copy.copy(func.__dict__)
    cpy.__kwdefaults__ = copy.copy(func.__kwdefaults__)

    return cpy


def resolve_function_typevars(
    func: FunctionType, parameters: MappedParameters
) -> FunctionType:
    func = copy_function(func)
    # TODO: add to __closure__ instead
    func.__globals__.update(
        {
            var if isinstance(var, str) else var.__name__: typ
            for var, typ in parameters.items()
        }
    )

    return func


def get_annotated_members(cls: Type) -> Dict[str, Any]:
    annotations = get_type_hints(cls)

    return annotations


def get_parameters(cls: Type) -> Tuple[TypeVar, ...]:
    try:
        return cls.__parameters__
    except AttributeError:
        return tuple()


# types in numba are passed as Functions. Need to uderstand if this is
# actually type and extract it.
# for python types extracting key.
# But for generic classes constructors we are returning function with special field set
def get_class(
    func: Union[Function, numba.core.types.functions.Dispatcher]
) -> Optional[Type]:
    if isinstance(func, Function):
        # TODO what if we have multiple templates defined?
        if len(func.templates) < 1:
            return None

        key = func.templates[0].key

        # import pdb; pdb.set_trace()
        if not isinstance(key, (type, TGenericAlias, _GenericAlias)):
            return None

        return key
    elif isinstance(func, numba.core.types.functions.Dispatcher):
        key = None
        try:
            key = func.key().__numba_class_type__
        except:
            return None

        return key

    return None


def resolve_type(typ: MemberType, parameters: MappedParameters) -> NType:
    args = []
    if isinstance(typ, TypeVar):
        typ = parameters[typ]

    if isinstance(typ, str):
        typ = parameters[typ]

    # import pdb; pdb.set_trace()
    if isinstance(typ, (TGenericAlias, _GenericAlias)):
        for arg in typ.__args__:
            args.append(resolve_type(arg, parameters))
        typ = typ.__origin__

    if isinstance(typ, NType):
        return typ

    return python_numba_type_map.construct(typ, args)  # type: ignore


def resolve_members(
    members: MembersDict, parameters: MappedParameters
) -> ResolvedMembersList:
    result = []
    for name, typ in members.items():
        result.append((name, resolve_type(typ, parameters)))

    return result
