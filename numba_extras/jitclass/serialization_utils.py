import sys
import weakref

import typing_extensions as typing
from typing import Dict, Type, Optional, Any, Callable, ClassVar, Mapping

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
        self.__original_name = name
        return super().__init__(f'{name}MetaType', (StructRefProxyMeta, ), dct)

    def _reduce_states(self):
        # import pdb; pdb.set_trace()
        return {"class_info": getattr(self, class_info_attr)}

    @classmethod
    def _rebuild(cls, class_info):
        proxy_meta = meta_from_class_info(class_info)
        # import pdb; pdb.set_trace()
        # proxy_cls = from_class_info(class_info)
        return proxy_meta

    def _reduce_class(self):
        return SerializableStructRefProxyMetaType

    def __reduce__(self, *args, **kwargs):
        return ReduceMixin.__reduce__(self, *args, **kwargs)

    def __box__(self, ty, mi):
        # instance = self.create({})
        # import pdb; pdb.set_trace()
        if ty not in self.__cache:
            self.__cache[ty] = type.__new__(self, self.__original_name, (StructRefProxy,), self.members)

        # instance = type.__new__(self, self.__original_name, (StructRefProxy,), self.members)
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
        return self(self.__original_name, members)


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


# class SerializableStructRefProxyMeta(ReduceMixin, StructRefProxyMeta):
#     def _reduce_states(self):
#         return {"class_info": getattr(self, class_info_attr)}

#     @classmethod
#     def _rebuild(cls, class_info):
#         proxy_cls = from_class_info(class_info)
#         return proxy_cls

#     def _reduce_class(self):
#         return SerializableStructRefProxyMeta


# this class implements type constructor, which is called in boxing procedure.
# it ables to restore generated class from class_info
# class SerializableBoxing(ReduceMixin):
#     def __init__(self, cls):
#         self.cls = cls

#     def __call__(self, ty, mi):
#         instance = super().__new__(self.cls)
#         instance._type = ty
#         instance._meminfo = mi

#         return instance

#     def _reduce_states(self):
#         cls = self.cls
#         return {"class_info": getattr(cls, class_info_attr)}

#     @classmethod
#     def _rebuild(cls, class_info: "ClassInfo"):
#         proxy_cls = from_class_info(class_info)
#         return cls(proxy_cls)


# The same as StructRefProxy, but with ability to be pickled/unpickled when created dynamically
# class SerializableStructRefProxy(ReduceMixin, structref.StructRefProxy):
#     def _reduce_states(self):
#         import pdb; pdb.set_trace()
#         return {'class_info': getattr(self, class_info_attr),
#                 'state': self.__dict__}

#     @classmethod
#     def _rebuild(cls, class_info, state):
#         proxy_cls = from_class_info(class_info)
#         # if not proxy_cls in cls.__cache:
#         #     raise RuntimeError('something went wrong')

#         # ref_class = cls.__cache[proxy_cls]
#         # instance = ref_class.__new__(ref_class)
#         proxy_cls.__dict__.update(state)

#         return proxy_cls

#     def _reduce_class(self):
#         return SerializableStructRefProxy

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
        assert proxy_cls not in cls.__cache
        cls.__cache[proxy_cls] = ref_cls
        ref_cls.__proxy_cls__ = proxy_cls
        # ref_cls.instance_type = ref_cls

    @classmethod
    def create(cls, name, proxy_cls, call):
        # import pdb; pdb.set_trace()
        cls_info = getattr(proxy_cls, class_info_attr)
        ref_cls = type(name, (cls,), {class_info_attr: cls_info})
        # import pdb; pdb.set_trace()
        ref_cls = structref.register(ref_cls)
        cls.bind(ref_cls, proxy_cls)

        return ref_cls

    # def __getattr__(self, name):
    #     # import pdb; pdb.set_trace()
    #     print('__getattr__', name)
    #     if name == 'instance_type' or name == '_code':
    #         import pdb; pdb.set_trace()
    #         print(name)
    #         return None

    #     return super().__getattr__(name)


# class meta_A:


# @jitclass
# class A(metaclass=meta_A):
#     # class_member: int
#     ...

# type(A) == meta_A

# A_proxy - python
# A_ref - StructRef

# meta_A_proxy = Python
# meta_A_ref - StructRef

# ls = typedList.empty_list(A_ref)

# @njit
# def foo(A):
#     a = A()