import sys
import weakref

import typing_extensions as typing
from typing import Dict, Type, Optional, Any, Callable, ClassVar, Mapping, Generic

from numba import types
from numba.experimental import structref
from numba.core.serialize import ReduceMixin as nReduceMixin
from numba.core.types import Callable

class_info_attr = "__numba_class_info__"

# def debug(func):
#     def impl(*args, **kwargs):
#         try:
#             return func(*args, **kwargs)
#         except Exception as e:
#             print(e)
#             import pdb; pdb.set_trace()

#     return impl

# nested types

# Since classes produced by jitclass are dynamically generated,
# it cannot be pickled/unpickled.
# In order to serialize/deserialize jitclass we save information about original class
# (module and name), so we can restore original class and if requried re-create
# StructRef and StructRefProxy classes

class ReduceMixin:
    def _reduce_states(self):
        raise NotImplementedError

    @classmethod
    def _rebuild(cls, **kwargs):
        raise NotImplementedError

    def _reduce_class(self):
        return self.__class__

    def __reduce__(self):
        return nReduceMixin.__reduce__(self)

# Save information about original class
class ClassInfo:
    def __init__(self, typ, params):
        self.module = typ.__module__
        self.qualname = typ.__qualname__
        self.params = params

    def get_class(self):
        module = sys.modules[self.module]
        typ = getattr(module, self.qualname)

        return typ


def from_class_info(class_info: ClassInfo):
    from numba_extras.jitclass.jitclass import jitclass

    proxy_cls = class_info.get_class()

    # if jitclass._SerializableStructRefProxy not in proxy_cls.__bases__:
    # if structref.StructRefProxy not in proxy_cls.__bases__:
    if StructRefProxy not in proxy_cls.__bases__:
        params = class_info.params._asdict()
        proxy_cls = jitclass(**params)(proxy_cls)

    return proxy_cls

def meta_from_class_info(class_info: ClassInfo):
    from numba_extras.jitclass.jitclass import jitclass

    proxy_cls = class_info.get_class()

    # if jitclass._SerializableStructRefProxy not in proxy_cls.__bases__:
    # if structref.StructRefProxy not in proxy_cls.__bases__:
    if StructRefProxy not in proxy_cls.__bases__:
        params = class_info.params._asdict()
        return jitclass.get_meta(proxy_cls, params)

    return type(proxy_cls)

# using ABCMeta as Meta for class inherited from type results in error:
# TypeError: descriptor '__subclasses__' of 'type' object needs an argument

class SerializableBoxing(ReduceMixin):
    def __init__(self, cls):
        self.cls = cls

    def __call__(self, ty, mi):
        import pdb; pdb.set_trace()
        return self.cls.__box__(ty, mi)

    def _reduce_states(self):
        # import pdb; pdb.set_trace()
        cls = self.cls
        return {"rebuilder": cls._reduce_class(), "state": cls._reduce_states()}

    @classmethod
    def _rebuild(cls, rebuilder, state):
        restored = rebuilder._rebuild(**state)
        return cls(restored)


class StructRefProxyMeta(type):
    def __init__(self, name, bases, dict):
        @property
        def _numba_type_(self):
            import pdb; pdb.set_trace()
            return self._type

        # import pdb; pdb.set_trace()
        # su = super()
        # # __init__ = su.__init__
        # try:
        # super().__init__(name, bases, dict)
        type.__init__(self, name, bases, dict)
        # super().__init__(self, name, bases, dict)
        # except Exception as e:
        #     import pdb; pdb.set_trace()
        #     print(e)
        self._numba_box_ = SerializableBoxing(self)
        # self._numba_type_ = _numba_type_
        # self.__slots__ = ('_type', '_meminfo')


class StructRefProxy:
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self._numba_box_ = SerializableBoxing(self)

    # pass
    # _type = None
    # _meminfo = None
    # pass
    # __slots__ = ('_type', '_meminfo')

    @property
    def _numba_type_(self):
        """Returns the Numba type instance for this structref instance.

        Subclasses should NOT override.
        """
        import pdb; pdb.set_trace()
        return self._type





from numba import njit

from numba_extras.jitclass.typing_utils import (
    _GenericAlias,
    get_annotated_members,
    get_parameters,
    resolve_members,
    resolve_function_typevars,
    MappedParameters,
    MembersDict,
    MethodsDict,
    ResolvedMembersList,
    NType,
)

# from numba_extras.jitclass.typing_utils import (
#     _GenericAlias,
#     get_annotated_members,
#     get_parameters,
#     resolve_type,
#     resolve_members,
#     get_class,
#     MappedParameters,
#     MembersDict,
#     MethodsDict,
#     ResolvedMembersList,
#     NType,
# )

from numba_extras.jitclass.overload_utils import (
    get_methods,
    wrap_and_jit,
    make_python_wrapper,
    make_property,
    make_function,
    overload_methods,
    make_constructor,
)

from numba_extras.jitclass.boxing import define_boxing

import numba
from numba.core.extending import overload, overload_method

# using ABCMeta as Meta for class inherited from type results in error:
# TypeError: descriptor '__subclasses__' of 'type' object needs an argument

# class StructRefProxyMT(StructRefProxy, StructRefProxyMeta):
#     pass

# In order to support boxing of StructRef we need to be able to pickle/unpickle
# constructor. Since constructor is a part of generated type, it cannot be
# pickled by default tools.
class SerializableStructRefProxyMetaType(ReduceMixin, StructRefProxyMeta):
    __cache = {}

    def __new__(cls, name, dct):
        return type.__new__(cls, f'{name}MetaType', (StructRefProxyMeta, ), dct)

    def __init__(self, name, dct):
        self._original_name = name
        return super().__init__(f'{name}MetaType', (StructRefProxyMeta, ), dct)

    def _reduce_states(self):
        # import pdb; pdb.set_trace()
        return {"class_info": getattr(self, class_info_attr)}

    @classmethod
    def _rebuild(cls, class_info):
        # import pdb; pdb.set_trace()
        proxy_meta = meta_from_class_info(class_info)
        # proxy_cls = from_class_info(class_info)
        return proxy_meta

    def _reduce_class(self):
        return SerializableStructRefProxyMetaType

    def __reduce__(self, *args, **kwargs):
        return ReduceMixin.__reduce__(self, *args, **kwargs)

    def __box__(self, ty, mi):
        # instance = self.create({})
        import pdb; pdb.set_trace()
        if ty not in self.__cache:
            self.__cache[ty] = type.__new__(self, self._original_name, (StructRefProxy,), self.members)

        # instance = type.__new__(self, self._original_name, (StructRefProxy,), self.members)
        instance = self.__cache[ty]

        instance._type = ty
        instance._meminfo = mi
        instance._numba_type_ = ty

        return instance

    @staticmethod
    def instance_box(self, ty, mi):
        instance = object.__new__(self)
        # import pdb; pdb.set_trace()

        instance._type = ty
        instance._meminfo = mi
        instance._numba_type_ = ty

        return instance

    # def __call__(self, *args, **kwargs):
    def __call__(self, name, members):
        # import pdb; pdb.set_trace()
        # instance = super().__call__(*args, **kwargs)
        # instance = super().__call__(name, (StructRefProxy,), members)
        self.members = members
        import pdb; pdb.set_trace()
        instance = self.ctor(self, name, (StructRefProxy,), members)
        # type.__init__(instance, name, (StructRefProxy,), members)
        StructRefProxyMeta.__init__(instance, name, (StructRefProxy,), members)
        # instance.__init__(name, (StructRefProxy,), members)

        # import pdb; pdb.set_trace()

        instance._reduce_states = StructRefProxySerializer._reduce_states.__get__(instance)
        instance._reduce_class = StructRefProxySerializer._reduce_class.__get__(instance)
        instance._rebuild = StructRefProxySerializer._rebuild
        instance.__reduce__ = StructRefProxySerializer.__reduce__.__get__(instance)
        instance.__box__ = self.instance_box.__get__(instance)
        setattr(instance, class_info_attr, getattr(self, class_info_attr))

        return instance

    def create(self, members):
        # return type.__new__(self, self.__name__, (StructRefProxy,), members)
        # return self(self.__name__, (StructRefProxy,), members)
        return self(self._original_name, members)

    @staticmethod
    def extract_members(cls, params):
        members = params.members

        if not members:
            members = get_annotated_members(cls)

        methods = get_methods(cls)
        if "__init__" not in methods:
            methods["__init__"] = lambda self: None

        if "__new__" in methods:
            new = methods["__new__"]
            if new is Generic.__new__ or new is object.__new__:
                del methods["__new__"]

        if "__new__" in methods:
            raise NotImplementedError('User defined __new__ is not supported')

        return members, methods

    def make_ref_cls(self, cls, mapped_parameters, class_info):
        members = self._orig_members
        resolved_methods = self._orig_methods

        resolved_members = resolve_members(members, mapped_parameters)
        if len(mapped_parameters):
            resolved_methods = {name: resolve_function_typevars(method, mapped_parameters) for name, method in resolved_methods.items()}

        ref_type = SerializableStructRef.create_(f'{self._original_name}_Ref', class_info)
        ref_cls = ref_type(resolved_members)

        for name, method in resolved_methods.items():
            overload_method(ref_type, name, strict=False)(lambda *args, **kwargs: method)

        resolved_methods = {name: njit(method) for name, method in resolved_methods.items()}
        properties = {member_name: make_property(cls.__name__, member_name) for member_name in members.keys()}

        instance_ctor = njit(make_constructor(resolved_methods.get('__init__'), ref_cls))

        all_members = {**resolved_methods, **properties}
        all_members.update({'__new__': lambda cls, *args, **kwargs: instance_ctor(*args, **kwargs)})

        return ref_cls, all_members


class StructRefProxyMetaType(SerializableStructRefProxyMetaType):
    def __new__(cls, cls_, params):
        class_info = {class_info_attr: ClassInfo(cls_, params)}
        return super().__new__(cls, cls_.__name__, class_info)

    def __init__(self, cls, params):
        class_info = getattr(self, class_info_attr)
        super().__init__(cls.__name__, {class_info_attr: class_info})

        members, methods = self.extract_members(cls, params)
        self._instance = None
        self._orig_members = members
        self._orig_methods = methods

        ref_meta = SerializableStructRef.create(f'{self.__name__}_Ref', self)
        self.ref_meta = ref_meta([])

        define_boxing(ref_meta, self)

        meta_ctor = njit(make_constructor(None, self.ref_meta))
        self.ctor = lambda *args, **kwargs: meta_ctor()

        ref_cls, all_members = self.make_ref_cls(cls, {}, class_info)

        self.ref_cls = ref_cls
        self._members = all_members
        self.__class_members = {}

    def get_instance(self):
        if self._instance is None:
            self._instance = type.__new__(self, self._original_name, (StructRefProxy,), self._members)

        return self._instance

    def __box__(self, ty, mi):
        instance = self.get_instance()
        instance._type = ty
        instance._meminfo = mi
        instance._numba_type_ = ty

        return instance

    def __call__(self):
        instance = super().__call__(self._original_name, self._members)
        SerializableStructRef.bind(type(self.ref_cls), instance)

        define_boxing(type(self.ref_cls), instance)

        return instance


class GenericAliasStructRefProxyMetaType(SerializableStructRefProxyMetaType):
    def __new__(cls, cls_, params, template_params):
        class_info = {class_info_attr: ClassInfo(cls_, params)}
        return super().__new__(cls, f'{cls_.__name__}GenericAlias', class_info)

    def __init__(self, cls, params, template_params):
        self._template_params = template_params
        class_info = getattr(self, class_info_attr)
        super().__init__(f'{cls.__name__}GenericAlias', {class_info_attr: class_info})

        members, methods = self.extract_members(cls, params)
        self._instance = None
        self._orig_members = members
        self._orig_methods = methods

        ref_meta = SerializableStructRef.create(f'{self.__name__}_GenericAliasRef', self)
        self.ref_meta = ref_meta([])

        define_boxing(ref_meta, self)

        meta_ctor = njit(make_constructor(None, self.ref_meta))
        self.ctor = lambda *args, **kwargs: meta_ctor()

        mapped_parameters = {param: typ for param, typ in zip(cls.__parameters__, template_params)}

        ref_cls, all_members = self.make_ref_cls(cls, mapped_parameters, class_info)

        self.ref_cls = ref_cls
        self._members = all_members
        self.__class_members = {}

    def get_instance(self):
        if self._instance is None:
            self._instance = type.__new__(self, self._original_name, (StructRefProxy,), self._members)

        return self._instance

    def __box__(self, ty, mi):
        instance = self.get_instance()
        instance._type = ty
        instance._meminfo = mi
        instance._numba_type_ = ty

        return instance

    def __call__(self):
        import pdb; pdb.set_trace()
        instance = super().__call__(self._original_name, self._members)
        SerializableStructRef.bind(type(self.ref_cls), instance)

        define_boxing(type(self.ref_cls), instance)

        return instance


class GenericStructRefProxyMetaType(SerializableStructRefProxyMetaType):
    def __new__(cls, cls_, params):
        class_info = {class_info_attr: ClassInfo(cls_, params), '__call__': cls.instance_call, '__getitem__': cls.instance_getitem}
        return super().__new__(cls, f'{cls_.__name__}Generic', class_info)

    def __init__(self, cls, params):
        class_info = getattr(self, class_info_attr)
        super().__init__(f'{cls.__name__}Generic', {class_info_attr: class_info, '__call__': self.instance_call, '__getitem__': self.instance_getitem})

        members, methods = self.extract_members(cls, params)
        self._instance = None
        # self._orig_members = members
        self._orig_members = {}
        # self._orig_methods = methods
        self._orig_methods = {}

        ref_meta = SerializableStructRef.create(f'{self.__name__}_GenericRef', self)
        self.ref_meta = ref_meta([])

        define_boxing(ref_meta, self)

        meta_ctor = njit(make_constructor(None, self.ref_meta))
        self.ctor = lambda *args, **kwargs: meta_ctor()

        ref_cls, all_members = self.make_ref_cls(cls, {}, class_info)

        self.ref_cls = ref_cls
        self._members = all_members
        import pdb; pdb.set_trace()
        self.cls = cls
        self.params = params
        # self.__class_members = {}
        # self._members = {}

    def get_instance(self):
        if self._instance is None:
            self._instance = type.__new__(self, self._original_name, (StructRefProxy,), self._members)

        return self._instance

    def __box__(self, ty, mi):
        instance = self.get_instance()
        instance._type = ty
        instance._meminfo = mi
        instance._numba_type_ = ty

        return instance

    def __call__(self):
        instance = super().__call__(self._original_name, self._members)
        SerializableStructRef.bind(type(self.ref_cls), instance)

        define_boxing(type(self.ref_cls), instance)

        return instance

    @staticmethod
    def instance_call(self, *args, **kwargs):
        raise RuntimeError

    @staticmethod
    def instance_getitem(self, args):
        if not isinstance(args, tuple):
            args = (args,)
        meta = GenericAliasStructRefProxyMetaType(self.cls, self.params, args)

        typ = meta()

        return _GenericAlias(typ, args)


# class GenericStructRefProxyMetaType(SerializableStructRefProxyMetaType):
#     @staticmethod
#     def extract_members(cls, params):
#         members = params.members

#         if not members:
#             members = get_annotated_members(cls)

#         methods = get_methods(cls)
#         if "__init__" not in methods:
#             methods["__init__"] = lambda self: None

#         if "__new__" in methods:
#             new = methods["__new__"]
#             if new is Generic.__new__ or new is object.__new__:
#                 del methods["__new__"]

#         if "__new__" in methods:
#             raise NotImplementedError('Non-default __new__ is not supported')

#         return members, methods

#     def __new__(cls, cls_, params):
#         class_info = {class_info_attr: ClassInfo(cls_, params)}
#         return super().__new__(cls, cls_.__name__, class_info)

#     def __init__(self, cls, params):
#         class_info = getattr(self, class_info_attr)
#         super().__init__(cls.__name__, {class_info_attr: class_info})

#         members, methods = self.extract_members(cls, params)
#         self.__instance = None
#         self._orig_members = members
#         self._orig_methods = methods

#         ref_meta = SerializableStructRef.create(f'{self.__name__}_Ref', self)
#         self.ref_meta = ref_meta([])

#         define_boxing(ref_meta, self)

#         meta_ctor = njit(make_constructor(None, self.ref_meta))
#         self.ctor = lambda *args, **kwargs: meta_ctor()

#         properties = {member_name: make_property(cls.__name__, member_name) for member_name in members.keys()}
#         all_members = {**methods, **properties}
#         resolved_members = resolve_members(members, {})

#         ref_cls = SerializableStructRef.create_(f'{self._original_name}_Ref', class_info)
#         self.ref_cls = ref_cls(resolved_members)
#         instance_ctor = njit(make_constructor(methods['__init__'], self.ref_cls))

#         all_members.update({'__new__': lambda cls, *args, **kwargs: instance_ctor(*args, **kwargs)})

#         self.__members = all_members
#         self.__class_members = {}

#     def get_instance(self):
#         if self.__instance is None:
#             self.__instance = type.__new__(self, self._original_name, (StructRefProxy,), self.__members)

#         return self.__instance

#     def __box__(self, ty, mi):
#         instance = self.get_instance()
#         instance._type = ty
#         instance._meminfo = mi
#         instance._numba_type_ = ty

#         return instance

#     def __call__(self):
#         # import pdb; pdb.set_trace()
#         instance = super().__call__(self._original_name, self.__members)
#         SerializableStructRef.bind(type(self.ref_cls), instance)

#         define_boxing(type(self.ref_cls), instance)

#         return instance

#     def _specificize_impl(args):
#         mapped_parameters = {param: typ for param, typ in zip(self.__parameters, args)}
#         members_list = resolve_members(self.__orig_members, mapped_parameters)
#         methods = {name: resolve_function_typevars}
#         ref_cls = SerializableStructRef.create_(f'{self._original_name}_Ref', class_info)

#     def specificize(self, args: List[Type]):
#         if len(args) != len(self.parameters):
#             msg = f"Wrong number of args. "
#             msg += f"Expected {len(self.parameters)}({self.parameters})."
#             msg += f"Got {len(args)}({args})"

#             raise RuntimeError(msg)

#         _args = tuple(args)
#         specificized = self.specificized.get(_args)

#         if not specificized:
#             specificized = self._specificize_impl(_args)
#             self.specificized[_args] = specificized

#         return specificized

#     def getitem_impl(self, *args):


#     def __getitem__(self, *args):
#         @njit
#         def getitem(self, *args):
#             return self[args]


class StructRefProxySerializer(ReduceMixin):
    def _reduce_states(self):
        # import pdb; pdb.set_trace()
        return {"class_info": getattr(self, class_info_attr)}

    @classmethod
    def _rebuild(cls, class_info):
        # import pdb; pdb.set_trace()
        proxy_cls = from_class_info(class_info)
        return proxy_cls

    def _reduce_class(self):
        return StructRefProxySerializer


# The same as StructRef, but with ability to be pickled/unpickled when created dynamically
class SerializableStructRef(ReduceMixin, types.StructRef):
    # CacheType = Mapping[structref.StructRefProxy, types.StructRef]
    CacheType = Mapping[StructRefProxy, types.StructRef]
    __cache: ClassVar[CacheType] = weakref.WeakKeyDictionary()

    def _reduce_states(self):
        state = self.__dict__
        # try:
        #     del state['instance_type']
        # except KeyError:
        #     pass

        proxy_cls = self.__proxy_cls__
        return {"rebuilder": proxy_cls._reduce_class(),
                "proxy_state": proxy_cls._reduce_states(),
                "state": state}
        # return {"class_info": getattr(self, class_info_attr), "state": self.__dict__}

    @classmethod
    def _rebuild(cls, rebuilder, proxy_state, state):
        # proxy_cls = from_class_info(class_info)
        proxy_cls = rebuilder._rebuild(**proxy_state)
        if not proxy_cls in cls.__cache:
            raise RuntimeError("something went wrong")

        ref_class = cls.__cache[proxy_cls]
        instance = ref_class.__new__(ref_class)
        instance.__dict__.update(state)
        # instance.instance_type = instance

        return instance

    def _reduce_class(self):
        return SerializableStructRef

    @classmethod
    def bind(cls, ref_cls, proxy_cls):
        import pdb; pdb.set_trace()
        assert proxy_cls not in cls.__cache
        cls.__cache[proxy_cls] = ref_cls
        ref_cls.__proxy_cls__ = proxy_cls
        # ref_cls.instance_type = ref_cls

    @classmethod
    def create(cls, name, proxy_cls, call = None):
        # import pdb; pdb.set_trace()
        cls_info = getattr(proxy_cls, class_info_attr)
        ref_cls = type(name, (cls,), {class_info_attr: cls_info})
        # import pdb; pdb.set_trace()
        ref_cls = structref.register(ref_cls)
        cls.bind(ref_cls, proxy_cls)

        return ref_cls

    @classmethod
    def create_(cls, name, class_info):
        ref_cls = type(name, (cls,), {class_info_attr: class_info})
        ref_cls = structref.register(ref_cls)

        return ref_cls