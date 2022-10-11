import subprocess
import importlib
from numba_extras import parallel_aot


def test_compile_add_example(tmp_path):
    tmp = tmp_path / "my_add"
    tmp.mkdir()

    p = tmp / 'add.py'
    content = "def add(a, b): return a + b"
    p.write_text(content)

    tup = (
        ("add", "int32(int32, int32)", "add_int32"),
        ("add", "int64(int64, int64)", "add_int64"),
        ("add", "float32(float32, float32)", "add_float32"),
        ("add", "float64(float64, float64)", "add_float64"),
    )
    for func, sig, func_name in tup:
        argv = [
            str(p),
            "emit-obj",
            "-f",
            func,
            "-n",
            func_name,
            "-s",
            sig,
            "-o",
            f"{(tmp / func_name).with_suffix('.o')}"
        ]
        parallel_aot.main(argv)

    argv = [
        f"{tmp / 'add.py'}",
        "merge",
        f"{tmp / 'add_int32.o'}",
        f"{tmp / 'add_int64.o'}",
        f"{tmp / 'add_float32.o'}",
        f"{tmp / 'add_float64.o'}",
        "-o",
        f"{tmp / 'add.so'}",
    ]
    parallel_aot.main(argv)

    cmd = ["ls"]
    ret = subprocess.run(
        cmd, cwd=str(tmp), capture_output=True, check=True
    )
    files = ret.stdout.decode("utf-8").split("\n")
    for filename in ["int32", "int64", "float64", "float64"]:
        assert f"add_{filename}.o" in files
        assert f"add_{filename}.pickle" in files

    assert "add.so" in files

    spec = importlib.machinery.PathFinder().find_spec("add", [str(tmp)])
    module = importlib.util.module_from_spec(spec)
    assert module.add_int64(2, 3) == 5
    assert module.add_float32(2.2, 3.3) == 5.5
