import cudf
import timeit
import sys
# NOTE: pandas with pyarrow backends produces worse results. Most likely because arithmetic
# is not implemented for pyarrow backends and the columns are then converted into numpy arrays.
task_init = timeit.default_timer()
x = cudf.read_csv(sys.argv[1])
print("init read, took %0.fs" % (timeit.default_timer()-task_init), flush=True)
exit(0)