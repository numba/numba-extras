import sys
import weakref

import typing_extensions as typing
from typing import Dict, Type, Optional, Any, Callable, ClassVar, Mapping

from numba import types
from numba.experimental import structref
from numba.core.serialize import ReduceMixin
from numba.core.types import Callable

class_info_attr = "__class_info"

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


class SerializableBoxing(ReduceMixin):
    def __init__(self, cls):
        self.cls = cls

    def __call__(self, ty, mi):
        instance = super().__new__(self.cls)
        instance._type = ty
        instance._meminfo = mi

        return instance

    def _reduce_states(self):
        cls = self.cls
        return {"class_info": getattr(cls, class_info_attr)}

    @classmethod
    def _rebuild(cls, class_info: "ClassInfo"):
        proxy_cls = from_class_info(class_info)
        return cls(proxy_cls)


class StructRefProxyMeta(type):
    def __init__(self, name, bases, dict):
        @property
        def _numba_type_(self):
            return self._type

        super().__init__(name, bases, dict)
        self._numba_box_ = SerializableBoxing(self)
        # self._numba_type_ = _numba_type_
        # self.__slots__ = ('_type', '_meminfo')


class StructRefProxy:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._numba_box_ = SerializableBoxing(self)

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
        return self._type

# class StructRefProxyMT(StructRefProxy, StructRefProxyMeta):
#     pass

# In order to support boxing of StructRef we need to be able to pickle/unpickle
# constructor. Since constructor is a part of generated type, it cannot be
# pickled by default tools.
class SerializableStructRefProxyMetaType(ReduceMixin, StructRefProxyMeta):
    def __new__(cls, name, bases, dct):
        dct.update({'__new__': lambda cls, name, bases, dct: type.__new__(name, bases, dct),
                    '__init__': lambda cls, name, bases, dct: type.__init__(name, bases, dct)})
        return type.__new__(cls, name, bases, dct)

    def _reduce_states(self):
        return {"class_info": getattr(self, class_info_attr)}

    @classmethod
    def _rebuild(cls, class_info):
        proxy_cls = from_class_info(class_info)
        return type(proxy_cls)

    def _reduce_class(self):
        return SerializableStructRefProxyMetaType

    def __call__(self, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        instance._reduce_states = StructRefProxySerializer._reduce_states.__get__(instance)
        instance._reduce_class = StructRefProxySerializer._reduce_class.__get__(instance)
        instance._rebuild = StructRefProxySerializer._rebuild
        instance.__reduce__ = StructRefProxySerializer.__reduce__.__get__(instance)
        setattr(instance, class_info_attr, getattr(self, class_info_attr))

        return instance


class StructRefProxySerializer(ReduceMixin):
    def _reduce_states(self):
        return {"class_info": getattr(self, class_info_attr)}

    @classmethod
    def _rebuild(cls, class_info):
        proxy_cls = from_class_info(class_info)
        return proxy_cls

    def _reduce_class(self):
        return StructRefProxySerializer


class SerializableStructRefProxyMeta(ReduceMixin, StructRefProxyMeta):
    def _reduce_states(self):
        return {"class_info": getattr(self, class_info_attr)}

    @classmethod
    def _rebuild(cls, class_info):
        proxy_cls = from_class_info(class_info)
        return proxy_cls

    def _reduce_class(self):
        return SerializableStructRefProxyMeta


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
        return {"class_info": getattr(self, class_info_attr), "state": self.__dict__}

    @classmethod
    def _rebuild(cls, class_info, state):
        proxy_cls = from_class_info(class_info)
        if not proxy_cls in cls.__cache:
            raise RuntimeError("something went wrong")

        ref_class = cls.__cache[proxy_cls]
        instance = ref_class.__new__(ref_class)
        instance.__dict__.update(state)

        return instance

    def _reduce_class(self):
        return SerializableStructRef

    @classmethod
    def bind(cls, ref_cls, proxy_cls):
        assert proxy_cls not in cls.__cache
        cls.__cache[proxy_cls] = ref_cls

    @classmethod
    def create(cls, name, proxy_cls, call):
        cls_info = getattr(proxy_cls, class_info_attr)
        ref_cls = type(name, (cls,), {class_info_attr: cls_info})
        # import pdb; pdb.set_trace()
        ref_cls = structref.register(ref_cls)
        cls.bind(ref_cls, proxy_cls)

        return ref_cls
