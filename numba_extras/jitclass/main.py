import numpy as np

import types as ptypes
from numba import njit
from numba.core import types, serialize
from numba import typed
from numba.experimental import structref
import typing
from typing import ClassVar, Generic, TypeVar, get_type_hints, List, Dict, Any, Optional, Tuple, Type
import numba
from numba.core.extending import overload, overload_method
from numba.core.types.functions import Function

import operator

from numba.tests.support import skip_unless_scipy

import inspect
import textwrap

python_numba_type_map = {int: lambda x: types.int64,
                         float: lambda x: types.float64,
                         str: lambda x: types.unicode_type,
                         List: lambda x: types.containers.ListType(x[0]),
                         list: lambda x: types.containers.ListType(x[0]),
                         Dict: lambda x: types.containers.DictType(x[0], x[1]),
                         dict: lambda x: types.containers.DictType(x[0], x[1])}

# available in typing module for python 3.8
def get_args(cls):
    return cls.__parameters__

def resolve_function_typevars(func, parameters):
    # TODO FIX it! Need to make function copy. Serialize/deserialize?
    func.__globals__.update({var.__name__: typ for var, typ in parameters.items()})
    return func

def make_function(name, sig, body, glbls, prefix = ' '*4):
    aligned_body = textwrap.indent(textwrap.dedent(body), prefix)

    sig_str = sig
    if isinstance(sig, inspect.Signature):
        sig_str = str(sig)

    sig_str = sig_str.replace('(', '').replace(')', '')

    definition = f'def {name}({sig_str}):\n'

    exec_glbls = glbls.copy()

    # print(definition + aligned_body)

    exec(definition + aligned_body, exec_glbls)

    return exec_glbls[name]

def get_methods(cls):
    methods = inspect.getmembers(cls, predicate=inspect.isfunction)

    return {name: value for name, value in methods}

def get_members(cls):
    annotations = get_type_hints(cls)

    return annotations

def get_parameters(cls):
    return get_args(cls)

def make_methods(methods):
    return {name: njit(make_python_wrapper(method)) for name, method in methods.items()}

def overload_methods(methods, struct_type_instance, struct_type, parameters):
    # overloading methods for generic classes works as follows:
    # python_wrapper->jit_wrapper->overload_impl(initial python method)

    # jit_wrapper simply calls to overloaded method. It is needed to create python_wrapper,
    # which must be created BEFORE template parameters are known. So actually jit_wrapper is needed
    # to do dispatching on argument types.

    # @overload mechanics allows to delay generation of implementation to the point when template
    # arguments are known. i.e. each time when type[args] is called we are generating new overload with
    # specific types
    for name, method in methods.items():
        sig_str = str(inspect.signature(method))
        sig_str = sig_str.replace('(', '').replace(')', '')
        impl = resolve_function_typevars(method, parameters)

        overload_impl = make_function(name + 'ovld', sig_str, f"""
            if self != struct_type_instance:
                return None

            return impl
        """, {'impl': impl, 'struct_type_instance': struct_type_instance})

        overload(method)(overload_impl) # to call from python wrapper

        # to call from jit-region
        if (name.startswith('__') and name.endswith('__')) and name[2:-2] in operator.__all__:
            overload(getattr(operator, name))(overload_impl)
        elif name == '__len__':
            # TODO for some reason overload(len) works only in module namespace, and doesn't work here
            # overload(len)(overload_impl)
            struct_type_instance.__numba_len_impl__ = impl
        else:
            overload_method(struct_type, name)(overload_impl)

# Workaround for issue with overload(len) declared in function
@overload(len)
def len_ovld(self):
    try:
        return self.__numba_len_impl__
    except:
        pass

    return None

# self, a, b => a, b, so can be safely forwarded into constructor call
def params_without_self(func):
    sig = inspect.signature(func)
    parameters = sig.parameters.copy()
    first_param = next(iter(parameters))
    del parameters[first_param]

    return sig.replace(parameters=parameters.values())

def make_constructor(init, struct_type, parameters):
    sig = params_without_self(init)
    args_str = str(sig)[1:-1] # remove trailing '(' and ')'
    frwd_args = ','.join(sig.parameters.keys())

    init = resolve_function_typevars(init, parameters)
    init = njit(init)
    body = f"""
        st = new(struct_type)
        init(st, {frwd_args})

        return st
    """

    glbls = {'struct_type': struct_type, 'init': init, 'new': structref.new}

    return make_function('ctor', args_str, body, glbls)

def get_name(cls):
    return cls.__name__

# given foo - function or CPUDispatcher
# =>
# def foo_wrapper(args):
#     return foo(args)
# creates python wrapper for python class for compiled methods
def make_python_wrapper(func):
    py_func = func
    try:
        py_func = func.py_func
    except:
        pass

    sig = inspect.signature(py_func)
    args_str = str(sig)[1:-1] # remove trailing '(' and ')'
    frwd_args = ','.join(sig.parameters.keys())
    func_name = py_func.__name__

    func_glbls = ({func_name: func})

    return make_function(f'{func_name}_wrapper', args_str, f'return {func_name}({frwd_args})', func_glbls)

def make_property(class_name, member_name):
    # TODO private members?
    get_name = f'{class_name}_get_{member_name}'
    set_name = f'{class_name}_set_{member_name}'

    get_jit = njit(make_function(get_name, 'self', f'return self.{member_name}', {}))
    set_jit = njit(make_function(set_name, 'self, value', f'self.{member_name} = value', {}))

    get_wrapper = make_python_wrapper(get_jit)
    set_wrapper = make_python_wrapper(set_jit)

    return property(get_wrapper).setter(set_wrapper)

# need to override behavior of typing._GenericAlias to pass
# templates argument into constructor in the following case:
# klass[int](...)
class _GenericAlias:
    def __init__(self, cls, params):
        self.__alias = typing._GenericAlias(cls, params)

    def __getitem__(self, params):
        return self.__alias.__getitem__(params)

    def __repr__(self):
        return self.__alias.__repr__()

    def __eq__(self, other):
        return self.__alias.__eq__(other)

    def __hash__(self):
        return self.__hash__()

    def __call__(self, *args, **kwargs):
        return self.__alias.__origin__.__initialize__(self.__alias.__args__, *args, **kwargs)

    def __mro_entries__(self, bases):
        return self.__alias.__mro_entries__(bases)

    def __getattr__(self, name):
        return getattr(self.__alias, name)

    def __setattr__(self, attr, val):
        if attr != '_GenericAlias__alias':
            return setattr(self.__alias, attr, val)
        else:
            return super().__setattr__(attr, val)

    def __instancecheck__(self, obj):
        return self.__subclasscheck__(type(obj))

    def __subclasscheck__(self, cls):
        return self.__alias.__subclasscheck__(cls)

    def __reduce__(self):
        return self.__alias.__reduce__()

class ClassDescriptor:

    def __init__(self, cls, compile_methods, members, jit_options):
        if not members:
            members = get_members(cls)

        jit_class_type = make_jit_class_type(cls)

        methods = get_methods(cls)
        ctor = None
        init = methods['__init__']
        if init:
            del methods['__init__']

        if '__new__' in methods:
            if methods['__new__'] is Generic.__new__:
                del methods['__new__']

        wrapped_methods = make_methods(methods)

        parameters = get_parameters(cls)

        # self is not properly initialized yet. Anything except capturing may result in wierd things
        python_class = make_class(get_name(cls), ctor, wrapped_methods, members, self)

        structref.define_boxing(jit_class_type, python_class)
        # TODO more args to 'init'
        overload(python_class)(make_function('init', 'self', "raise NotImplementedError('Not implemented')", {}))

        self.init = init
        self.numba_type = jit_class_type
        self.python_type = python_class
        self.parameters = parameters
        self.members = members
        self.wrapped_methods = wrapped_methods
        self.original_methods = methods
        self.specificized = {}

    def __resolve_members(self, mapped_parameters):
        members = [(key, value) for key, value in self.members.items()]
        return resolve_members(members, mapped_parameters)

    def __map_parameters(self, args):
        return {param: typ for param, typ in zip(self.parameters, args)}

    def __instantiate_struct_type(self, members, mapped_parameters):
        struct_type = self.numba_type
        struct_type_instance = struct_type(members)

        overload_methods(self.original_methods, struct_type_instance, struct_type, mapped_parameters)

        return struct_type_instance

    def __make_constructor(self, struct_type, mapped_parameters):
        ctor_impl = njit(make_constructor(self.init, struct_type, mapped_parameters))

        ctor_impl.__numba_class_type__ = struct_type

        return ctor_impl

    def __specificize(self, args):
        mapped_parameters = self.__map_parameters(args)
        members = self.__resolve_members(mapped_parameters)
        struct_type = self.__instantiate_struct_type(members, mapped_parameters)
        ctor_impl = self.__make_constructor(struct_type, mapped_parameters)

        return ctor_impl

    def specificize(self, args):
        if len(args) != len(self.parameters):
            msg = f'Wrong number of args. '
            msg += f'Expected {len(self.parameters)}({self.parameters}).'
            msg += f'Got {len(args)}({args})'

            raise RuntimeError(msg)

        args = tuple(args)
        specificized = self.specificized.get(args)

        if not specificized:
            specificized = self.__specificize(args)
            self.specificized[args] = specificized

        return specificized

# creates python proxy for our jitclass
def make_class(name, ctor, methods, members, class_descr):
    members = {member_name: make_property(name, member_name) for member_name in members.keys()}
    all_members = {}
    all_members = {method_name: make_python_wrapper(method) for method_name, method in methods.items()}
    all_members.update(members)
    all_members.update({'__new__': lambda cls: object.__new__(cls)}) # TODO figure out if we need it?

    def init(self, *args, **kwargs): # make valid args
        raise NotImplementedError('Not implemented')

    @classmethod
    def __class_getitem__(cls, params):
        alias = _GenericAlias(cls, params)

        return alias

    @classmethod
    def initializer(cls, params, *args, **kwargs):
        specifizied = class_descr.specificize(params)

        return specifizied(*args, **kwargs)

    all_members.update({'__init__': init,
                        '__class_getitem__': __class_getitem__,
                        '__initialize__': initializer})

    return type(name, (structref.StructRefProxy, ), all_members)

__unique_struct_id = 0

# @structref.register
# class StructType(types.StructRef):

#     __ctor = njit(lambda types, *args, **kwargs: StructType[types](*args, **kwargs))

#     def __new__(cls):
#         return None

#     @classmethod
#     def _initialize(types, *args, **kwargs):
#         return ctor(types, *args, **kwargs)

def make_jit_class_type(cls):
    global __unique_struct_id
    struct_name = f'__StructType_{__unique_struct_id}'
    struct_type = type(struct_name, (types.StructRef,), {})
    # TODO need to pickle/unpickle struct_type. Need to find more apropriate way to do it
    struct_type.__module__ = __name__
    import sys
    module = sys.modules[__name__]
    setattr(module, struct_name, struct_type)
    __unique_struct_id += 1

    typ = getattr(module, struct_name)
    typ = structref.register(typ)
    setattr(module, struct_name, typ)

    return typ
    # return StructType

# types in numba are passed as Functions. Need to uderstand if this is
# actually type and extract it.
# for python types extracting key.
# But for generic classes constructors we are returning function with special field set
def get_class(func):
    if isinstance(func, Function):
        # TODO what if we have multiple templates defined?
        if len(func.templates) < 1:
            return None

        key = func.templates[0].key

        if not isinstance(key, (type, typing._GenericAlias)):
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

def resolve_type(typ, parameters):
    args = []
    if isinstance(typ, TypeVar):
        typ = parameters[typ]
    elif isinstance(typ, typing._GenericAlias):
        for arg in typ.__args__:
            args.append(resolve_type(arg, parameters))
        typ = typ.__origin__

    if typ in python_numba_type_map:
        typ = python_numba_type_map[typ](args)

    return typ

def resolve_members(members, parameters):
    result = []
    for name, typ in members:
        result.append((name, resolve_type(typ, parameters)))

    return result

class jitclass:
    ClassTypes = Dict[type, ClassDescriptor]
    Members = Optional[Dict[str, type]]
    Options = Optional[Dict[str, Any]]

    __class_types: ClassVar[ClassTypes] = {}

    compile_methods: bool
    jit_options:     Options
    members:         Members

    def __new__(cls:             Type,
                *args:           Type,
                compile_methods: bool    = True,
                jit_options:     Options = None,
                members:         Members = None):
        if len(args):
            if len(args) > 1:
                raise RuntimeError('Something went wrong')

            origin_class = args[0]
            if not isinstance(origin_class, type):
                raise RuntimeError('Something went wrong')

            return jitclass(compile_methods=compile_methods, jit_options=jit_options, members=members)(origin_class)

        return super().__new__(cls)

    def __init__(self:            Any,
                 *args:           Type,
                 compile_methods: bool    = True,
                 jit_options:     Options = None,
                 members:         Members = None):
        if not isinstance(self, jitclass):
            return None

        if len(args):
            raise RuntimeError('Something went wrong')

        self.compile_methods = compile_methods
        self.jit_options = jit_options
        self.members = members

    def __call__(self, cls=None):
        class_descr = ClassDescriptor(cls, self.compile_methods, self.members, self.jit_options)
        python_class = class_descr.python_type
        jitclass.__class_types[python_class] = class_descr

        return python_class

    @classmethod
    @overload(operator.getitem)
    def __class_getitem(cls, args):
        _cls = get_class(cls)
        if _cls is None:
            return None

        if not isinstance(args, (types.containers.Tuple, types.containers.UniTuple)):
            args = (args, )

        args = [get_class(arg) for arg in args]
        if not all(args):
            # TODO raise an error
            return None

        if _cls is List:
            typ = resolve_type(args[0], {})

            @njit
            def ctor_impl():
                return typed.List.empty_list(typ)

            ctor_impl.__numba_class_type__ = python_numba_type_map[_cls]([typ])

            def ovl_impl(cls, args):
                return ctor_impl

            return ovl_impl

        if _cls is Dict:
            key_type = resolve_type(args[0], {})
            value_type = resolve_type(args[1], {})

            @njit
            def ctor_impl():
                return typed.Dict.empty(key_type=key_type, value_type=value_type)

            ctor_impl.__numba_class_type__ = python_numba_type_map[_cls]([key_type, value_type])

            def ovl_impl(cls, args):
                return ctor_impl

            return ovl_impl

        class_descr = jitclass.__class_types.get(_cls)

        if class_descr is None:
            # TODO raise an error
            return None

        ctor_impl = class_descr.specificize(args)

        def ovl_impl(cls, args):
            return ctor_impl

        return ovl_impl

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

T = TypeVar('T')

@jitclass
class bar(Generic[T]):
    a: List[T]

    def __init__(self, v1, v2 = 1):
        self.a = List[T]()

        self.a.append(v1)
        self.a.append(v2)

    def append(self, value):
        self.a.append(value)

    def __getitem__(self, index):
        return self.a[index]

    def __setitem__(self, index, value):
        self.a[index] = value

    def __len__(self):
        return len(self.a)

# @overload(len)
# def len_impl(self):
#     # import pdb; pdb.set_trace()
#     def impl(self):
#         return len(self.a)

#     return impl

@njit
def foo():
    b = bar[int](0)

    b.append(1)

    print(len(b))
    b[0] = 100500
    print(b[0])

    d = Dict[int, str]()

    # return d
    return b, d

b, d = foo()

b.append(10)
print(b.a)

print(d)

a = bar[int](-1)
a.append(3)
print(a.a)

c = bar[float](-2)
print(c.a)
print(len(c))
c[0] = 100

print(c[0])
