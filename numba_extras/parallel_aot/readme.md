## Intro

The Numba project contains a package for an [ahead-of-time compiler](https://numba.pydata.org/numba-doc/dev/user/pycc.html). It works by annotating the exported function with a decorator `@cc.export(fn_name, signature)`

```python
from numba.pycc import CC
cc = CC('my_module')

@cc.export('multf', 'f8(f8, f8)')
@cc.export('multi', 'i4(i4, i4)')
def mult(a, b):
    return a * b

@cc.export('square', 'f8(f8)')
def square(a):
    return a ** 2

if __name__ == "__main__":
    cc.compile()
```

This work originates from https://github.com/numba/numba/issues/6424 and adds a python script that server as an entry point for the Numba AOT compiler. The script enables Numba to compile functions ahead of time from the command line. As a result, one can now use GNU make to compile functions in parallel for different signatures (see quicksort example below). 

## How to use it

```python
# add.py
def add(a, b):
    return a + b

```

Then, run

```bash
# create object files
$ python numba-aot.py add.py emit-obj -f 'add' -n 'addi' -s 'int64(int64, int64)' -o 'addi.o'
$ python numba-aot.py add.py emit-obj -f 'add' -n 'addf' -s 'float64(float64, float64)' -o 'addf.o'

# emit shared library
$ python numba-aot.py add.py merge addi.o addf.o
```

Or via `Makefile`

```makefile
# via Makefile
targets := int32 int64 float32 float64

all: $(targets)
	numba add.py merge *.o

run:
	python -c "import my_module; print(dir(my_module))"

$(targets):
	python numba-aot.py add.py emit-obj -f add -n add_$@ -s "$@($@, $@)" -o add_$@.o

clean:
	rm -f *.o *.pickle *.so
```

More examples can be found in the `examples/` folder.

## Speedup

The quicksort example compiles in 19s using 20 jobs with a makefile.

```bash
$ time make -j20
...
make -j20  313.97s user 63.04s system 1900% cpu 19.842 total
```

The single thread version takes 81 seconds to compile

```bash
$ time python quicksort.py
python quick.py  81.62s user 1.78s system 102% cpu 1:21.24 total
```

## Caveats

To build the shared library, Numba needs to record each exported entry alongside its LLVM function and environment info. To make it work with parallel AOT compilation, this information is serialized to a second file with the same name and extension `.pickled` (i.e. `addi.pickled`) when generating `*.o`.

## Limitations

All the original [limitations](https://numba.pydata.org/numba-doc/dev/user/pycc.html#limitations) persist.