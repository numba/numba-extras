targets := int32 int64 float32 float64

all: $(targets)
	python ../../numba-aot.py add.py merge /tmp/my_add/*.o -o /tmp/my_add/add.so

run:
	cd /tmp/my_add && python -c "import add; print(dir(add))"

$(targets):
	python ../../numba-aot.py add.py emit-obj -f add -n add_$@ -s "$@($@, $@)" -o /tmp/my_add/add_$@.o

clean:
	cd /tmp/my_add && rm -f *.o *.pickle *.so