import inspect
from inspect import Signature

import textwrap
import typing_extensions as typing
from typing import Union, Iterable, Optional, Dict, Callable

import operator

import numba
from numba import njit
from numba.core import types
from numba.experimental import structref
from numba.core.extending import overload, overload_method

from numba_extras.jitclass.typing_utils import resolve_function_typevars


def get_methods(cls):
    methods = inspect.getmembers(cls, predicate=inspect.isfunction)

    return {name: value for name, value in methods}


def params_without_self(func):
    sig = inspect.signature(func)
    parameters = sig.parameters.copy()
    first_param = next(iter(parameters))
    del parameters[first_param]

    return sig.replace(parameters=parameters.values())


SigOrStr = Union[str, Signature]


def sig_to_str(sig: SigOrStr) -> str:
    if isinstance(sig, Signature):
        sig = str(sig)

    if sig.startswith("(") and sig.endswith(")"):
        sig = sig[1:-1]

    return sig


def sig_to_args(sig: SigOrStr) -> Iterable:
    if isinstance(sig, str):
        sig = inspect.signature(make_function("func", sig, "pass"))

    return sig.parameters.keys()


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
    args_str = sig_to_str(sig)
    frwd_args = ",".join(sig_to_args(sig))
    func_name = py_func.__name__

    func_glbls = {func_name: func}

    return make_function(
        f"{func_name}_wrapper", args_str, f"return {func_name}({frwd_args})", func_glbls
    )


def wrap_and_jit(methods):
    from numba_extras.jitclass.jitclass import method_option

    result = {}
    for name, method in methods.items():
        if method_option(method).compile is False:
            result[name] = method
        else:
            result[name] = njit(make_python_wrapper(method))

    return result


def make_function(
    name: str,
    sig: SigOrStr,
    body: str,
    glbls: Optional[Dict] = None,
    prefix: str = " " * 4,
) -> Callable:
    if glbls is None:
        glbls = {}

    aligned_body = textwrap.indent(textwrap.dedent(body), prefix)

    sig_str = sig_to_str(sig)
    definition = f"def {name}({sig_str}):\n"

    exec_glbls = glbls.copy()
    exec(definition + aligned_body, exec_glbls)

    return exec_glbls[name]


def _make_overload(method):
    sig_str = sig_to_str(inspect.signature(method))
    # impl = resolve_function_typevars(method, parameters)

    overload_impl = make_function(
        method.__name__ + "ovld",
        sig_str,
        f"""
        return resolve_function_typevars(method, self.mapped_parameters)
    """,
        {"method": method, "resolve_function_typevars": resolve_function_typevars},
    )

    return overload_impl


def _wrap_overload(method):
    sig_str = sig_to_str(inspect.signature(method))
    frwd_args = ",".join(sig_to_args(sig_str))

    overload_impl = make_function(
        method.__name__ + "ovld",
        sig_str,
        f"""
        ovld = method({frwd_args})
        return resolve_function_typevars(ovld, self.mapped_parameters)
    """,
        {"method": method, "resolve_function_typevars": resolve_function_typevars,},
    )

    return overload_impl


def make_overload(method):
    from numba_extras.jitclass.jitclass import method_option

    if method_option(method).overload:
        return _wrap_overload(method)

    return _make_overload(method)


def overload_methods(methods, struct_type):
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
        sig_str = sig_str.replace("(", "").replace(")", "")
        # impl = resolve_function_typevars(method, parameters)

        # overload_impl = make_overload(method, parameters)
        overload_impl = make_overload(method)
        overload(method)(overload_impl)  # to call from python wrapper

        # to call from jit-region
        if (name.startswith("__") and name.endswith("__")) and name[
            2:-2
        ] in operator.__all__:
            overload(getattr(operator, name))(overload_impl)
        else:
            overload_method(struct_type, name)(overload_impl)


# Workaround for issue with overload(len) declared in function
@overload(len)
def len_ovld(self):
    try:
        return resolve_function_typevars(
            self.__numba_len_impl__, self.mapped_parameters
        )
    except:
        pass

    return None


def make_constructor(init, struct_type):
    # import pdb; pdb.set_trace()
    if init is not None:
        sig = params_without_self(init)
        args_str = sig_to_str(sig)
        frwd_args = ",".join(sig.parameters.keys())

        body = f"""
            st = new(struct_type)
            init(st, {frwd_args})

            return st
        """

        glbls = {"struct_type": struct_type, "init": init, "new": structref.new}

        return make_function("ctor", args_str, body, glbls)
    else:
        args_str = sig_to_str(inspect.signature(lambda: None))
        body = f"""
            return new(struct_type)
        """

        glbls = {"struct_type": struct_type, "new": structref.new}

        return make_function("ctor", args_str, body, glbls)


def make_property(class_name, member_name):
    # TODO private members?
    get_name = f"{class_name}_get_{member_name}"
    set_name = f"{class_name}_set_{member_name}"

    get_jit = njit(make_function(get_name, "self", f"return self.{member_name}", {}))
    set_jit = njit(
        make_function(set_name, "self, value", f"self.{member_name} = value", {})
    )

    return property(get_jit).setter(set_jit)
