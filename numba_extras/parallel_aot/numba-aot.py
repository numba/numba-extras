import os
import sys
import errno
import pickle
import tempfile
import importlib
import pathlib
import argparse

from distutils import log

from numba.pycc import CC
from numba.pycc.compiler import ModuleCompiler
from numba.core.compiler import Flags, compile_extra
from numba.core.runtime import nrtdynmod
from numba.core.compiler_lock import global_compiler_lock

import llvmlite.llvmpy.core as lc
from llvmlite.binding import Linkage


class ParallelModuleCompiler(ModuleCompiler):
    def __init__(
        self,
        export_entries,
        module_name,
        use_nrt=False,
        external_init_function=None,
        filename=None,
        **aot_options,
    ):
        super().__init__(export_entries, module_name, use_nrt, **aot_options)

        self.filename = filename
        self.external_init_function = external_init_function

        self.exported_function_types = {}
        self.function_environments = {}
        self.environment_gvs = {}

        self.codegen = self.context.codegen()
        self.library = self.codegen.create_library(module_name)

    def _emit_nrt_module(self):
        flags = Flags()
        flags.no_compile = True
        flags.nrt = True
        nrt_module, _ = nrtdynmod.create_nrt_module(self.context)
        self.library.add_ir_module(nrt_module)

    @global_compiler_lock
    def _cull_exports(self):
        """Read all the exported functions/modules in the translator
        environment, and join them into a single LLVM module.
        """
        # Generate IR for all exported functions
        flags = Flags()
        flags.no_compile = True
        if self.use_nrt:
            flags.nrt = True
            # Compile NRT helpers
            nrt_module, _ = nrtdynmod.create_nrt_module(self.context)
            self.library.add_ir_module(nrt_module)

        # parallel compilation for a single translation unit
        assert len(self.export_entries) == 1

        for entry in self.export_entries:
            function = entry.function
            entry.function = None
            cres = compile_extra(
                self.typing_context,
                self.context,
                function,
                entry.signature.args,
                entry.signature.return_type,
                flags,
                locals={},
                library=self.library,
            )

            func_name = cres.fndesc.llvm_func_name
            llvm_func = cres.library.get_function(func_name)

            # if self.export_python_wrap:
            llvm_func.linkage = lc.LINKAGE_INTERNAL
            wrappername = cres.fndesc.llvm_cpython_wrapper_name
            wrapper = cres.library.get_function(wrappername)
            wrapper.name = self._mangle_method_symbol(entry.symbol)
            wrapper.linkage = lc.LINKAGE_EXTERNAL
            fnty = cres.target_context.call_conv.get_function_type(
                cres.fndesc.restype, cres.fndesc.argtypes
            )
            self.exported_function_types[entry] = fnty
            self.function_environments[entry] = cres.environment
            self.environment_gvs[entry] = cres.fndesc.env_name

        if self.export_python_wrap:
            wrapper_module = self.library.create_ir_module("wrapper")
            self._emit_python_wrapper(wrapper_module)
            self.library.add_ir_module(wrapper_module)

        if not self.export_python_wrap:
            d = {
                "exported_function_types": self.exported_function_types,
                "function_environments": self.function_environments,
                "environment_gvs": self.environment_gvs,
                "export_entries": self.export_entries,
            }
            with pathlib.Path(self.filename).with_suffix(".pickle").open("wb") as f:
                pickle.dump(d, f)

        # Hide all functions in the DLL except those explicitly exported
        self.library.finalize()
        for fn in self.library.get_defined_functions():
            if fn.name not in self.dll_exports:
                if fn.linkage in {Linkage.private, Linkage.internal}:
                    # Private/Internal linkage must have "default" visibility
                    fn.visibility = "default"
                else:
                    fn.visibility = "hidden"
        return self.library


class ParallelCC(CC):
    def __init__(self, extension_name, source_module=None, output_dir=None):
        super().__init__(extension_name, source_module)
        # need to record the output dir for the merge step
        self._output_dir = output_dir

    @global_compiler_lock
    def emit_object_file(self, filename):
        """
        Compile the extension module into LLVM IR
        """
        compiler = ParallelModuleCompiler(
            self._export_entries,
            self._basename,
            use_nrt=False,  # self._use_nrt,
            cpu_name=self._target_cpu,
            filename=filename,
        )
        compiler.external_init_function = self._init_function
        temp_obj = str(pathlib.Path(filename).resolve())
        log.info("generating LLVM code for '%s' into %s", self._basename, temp_obj)
        compiler.write_native_object(temp_obj, wrap=False)

    def _compile_python_wrapper(self, files, build_dir):
        compiler = ParallelModuleCompiler(
            self._export_entries,
            self._basename,
            use_nrt=False,  # self._use_nrt,
            cpu_name=self._target_cpu,
            external_init_function=self._init_function,
        )
        compiler._emit_nrt_module()

        def get_path(file):
            path = pathlib.Path(file)
            if not path.exists():
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file)
            return str(path)

        objects = []
        for file in files:
            objects += [get_path(file)]

            # read pickle information
            with pathlib.Path(file).with_suffix(".pickle").resolve().open("rb") as f:
                pickled = pickle.load(f)

            compiler.exported_function_types = {
                **compiler.exported_function_types,
                **pickled["exported_function_types"],
            }
            compiler.function_environments = {
                **compiler.function_environments,
                **pickled["function_environments"],
            }
            compiler.environment_gvs = {
                **compiler.environment_gvs,
                **pickled["environment_gvs"],
            }
            compiler.export_entries += pickled["export_entries"]

        wrapper_mod = compiler.library.create_ir_module("wrapper")
        compiler._emit_python_wrapper(wrapper_mod)
        compiler.library.add_ir_module(wrapper_mod)

        temp_dir = tempfile.mkdtemp(prefix="pycc-build-%s-" % self._basename)
        temp_obj = os.path.join(temp_dir, "wrapper.o")
        with open(temp_obj, "wb") as fout:
            fout.write(compiler.library.emit_native_object())
        return objects + [temp_obj], compiler.dll_exports

    @global_compiler_lock
    def merge_object_files(self, files):
        objects, dll_exports = self._compile_python_wrapper(files, self._output_dir)

        temp_dir = tempfile.mkdtemp(prefix="pycc-build-%s-" % self._basename)
        objects += self._compile_mixins(temp_dir)

        extra_ldflags = self._get_extra_ldflags()
        output_dll = os.path.join(self._output_dir, self._output_file)
        libraries = self._toolchain.get_python_libraries()
        library_dirs = self._toolchain.get_python_library_dirs()
        self._toolchain.link_shared(
            output_dll,
            objects,
            libraries,
            library_dirs,
            export_symbols=dll_exports,
            extra_ldflags=extra_ldflags,
        )


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename", action="store", nargs="?", help="Python source filename"
    )
    sub_parsers = parser.add_subparsers(help="Numba AOT help", dest="kind")
    parser_llvm = sub_parsers.add_parser("emit-obj", help="emit object file")
    parser_merge = sub_parsers.add_parser("merge", help="merge object files (*.o)")

    parser_llvm.add_argument(
        "-f",
        "--function",
        action="store",
        type=str,
        required=True,
        help="The function to be exported",
    )

    parser_llvm.add_argument(
        "-n",
        "--name",
        action="store",
        type=str,
        required=True,
        help="Name of the exported function",
    )

    parser_llvm.add_argument(
        "-s",
        "--signature",
        action="store",
        type=str,
        required=True,
        help="Signature of the exported function",
    )

    parser_llvm.add_argument(
        "-o", action="store", type=str, required=True, help="Name of the output file"
    )

    parser_merge.add_argument(
        action="store",
        type=str,
        nargs="+",
        dest="files",
        help="list of llvm IR files to be merged",
    )
    parser_merge.add_argument(
        "-o",
        action="store",
        type=str,
        required=False,
        default="my_module",
        help="Name of the library file",
    )

    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    if args.kind is not None:
        # import module programatically
        # https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
        mod_name = args.filename.strip(".py")
        mod_path = os.path.abspath(args.filename)
        spec = importlib.util.spec_from_file_location(mod_name, mod_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)

        if args.kind == "emit-obj":
            fn_name = args.function
            exported_name = args.name
            sig = args.signature
            try:
                fn = getattr(module, fn_name)
            except AttributeError:
                raise ImportError(f"function {fn_name} not found in {module.__name__}")
            p = pathlib.Path(args.o)
            cc = ParallelCC("my_module", output_dir=p.parent)
            cc.export(exported_name, sig)(fn)
            cc.emit_object_file(args.o)
        elif args.kind == "merge":
            p = pathlib.Path(args.o)
            cc = ParallelCC(p.stem, output_dir=p.parent)
            cc.merge_object_files(args.files)
        else:
            raise RuntimeError


if __name__ == "__main__":
    main()
