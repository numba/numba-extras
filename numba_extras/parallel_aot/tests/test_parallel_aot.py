import subprocess
import importlib


def test_compile_add_example():
    subprocess.run(
        ["make", "clean"], cwd="numba_extras/parallel_aot/examples/add", check=True
    )
    subprocess.run(
        ["make", "-j10"], cwd="numba_extras/parallel_aot/examples/add", check=True
    )

    cmd = ["ls"]
    ret = subprocess.run(cmd, cwd="/tmp/my_add", capture_output=True, check=True)
    files = ret.stdout.decode("utf-8").split("\n")
    for filename in ["int32", "int64", "float32", "float64"]:
        assert f"add_{filename}.o" in files
        assert f"add_{filename}.pickle" in files

    assert "add.so" in files

    spec = importlib.machinery.PathFinder().find_spec("add", ["/tmp/my_add"])
    module = importlib.util.module_from_spec(spec)
    assert module.add_int64(2, 3) == 5
    assert module.add_float32(2.2, 3.3) == 5.5
