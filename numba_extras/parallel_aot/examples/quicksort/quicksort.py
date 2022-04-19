from __future__ import absolute_import, division, print_function

import numba
import numba.pycc
import numpy as np

# cc = numba.pycc.CC('extension')

# An iterative quicksort implementation which sorts multiple arrays doing row-wise comparisons.
# It uses numba and code generation to generate efficient code.
# The quicksort implementation in this file is a slightly adapted version of numpy's quicksort:
# https://github.com/numpy/numpy/blob/master/numpy/core/src/npysort/quicksort.c.src
_quicksort_template = '''
def quicksort(idxs, s, e, stack, {arr_names}):
    def lt(x, y):
        return {lt_impl_x_y}

    def swap(x, y):
        t = idxs[x]
        idxs[x] = idxs[y]
        idxs[y] = t

    stack[0] = s
    stack[1] = e - 1
    sptr = 2

    SMALL_QUICKSORT = 15

    while sptr > 0:
        # pop interval to be sorted from stack
        pr = stack[sptr - 1]
        pl = stack[sptr - 2]
        sptr -= 2

        # use quicksort iteratively until interval is too small
        while pr - pl > SMALL_QUICKSORT:
            # compute median element of left, median, right and swap
            pm = pl + ((pr - pl) >> 1)
            if lt(idxs[pm], idxs[pl]):
                swap(pm, pl)
            if lt(idxs[pr], idxs[pm]):
                swap(pr, pm)
            if lt(idxs[pm], idxs[pl]):
                swap(pm, pl)

            # this is our pivot value
            vp = idxs[pm]

            # stash pivot in last unknown element of interval (pr has to be greater than pm from above)
            pi = pl
            pj = pr - 1
            swap(pm, pj)

            # swap left and right according to pivot
            while True:
                pi += 1
                while lt(idxs[pi], vp):
                    pi += 1
                pj -= 1
                while lt(vp, idxs[pj]):
                    pj -= 1
                if pi >= pj:
                    break
                swap(pi, pj)

            # put pivot in its place
            pk = pr - 1
            swap(pi, pk)

            # push largest partition on stack
            assert sptr + 2 <= len(stack)
            if pi - pl < pr - pi:
                stack[sptr] = pi + 1
                stack[sptr + 1] = pr
                sptr += 2
                pr = pi - 1
            else:
                stack[sptr] = pl
                stack[sptr + 1] = pi - 1
                sptr += 2
                pl = pi + 1

        # insertion sort
        for pi in range(pl + 1, pr + 1):
            vp = idxs[pi]
            pj = pi
            pk = pi - 1
            while pj > pl and lt(vp, idxs[pk]):
                idxs[pj] = idxs[pk]
                pj -= 1
                pk -= 1
            idxs[pj] = vp
'''


def _get_quicksort(n_arrs):
    arr_names = tuple('a{}'.format(i) for i in range(n_arrs))
    lt_impl_x_y = '{an}[x] < {an}[y]'.format(an=arr_names[-1])
    for an in reversed(arr_names[:-1]):
        lt_impl_x_y = 'True if {an}[x] < {an}[y] else (False if {an}[x] > {an}[y] else ({lt_impl_x_y}))' \
            .format(an=an, lt_impl_x_y=lt_impl_x_y)
    quicksort_code = _quicksort_template.format(lt_impl_x_y=lt_impl_x_y, arr_names=', '.join(arr_names))
    quicksort_module = compile(quicksort_code, __file__, 'exec')
    # globals_dict = {}
    # locals_dict = {}
    # exec(quicksort_module, globals_dict, locals_dict)
    # return locals_dict['quicksort']
    exec(quicksort_module)
    return locals()['quicksort']

AOT_SORT_SUPPORTED_DTYPES = [np.dtype('int64'), np.dtype('int32'), np.dtype('float64')]

quicksort_1 = _get_quicksort(1)
quicksort_2 = _get_quicksort(2)
quicksort_3 = _get_quicksort(3)
quicksort_4 = _get_quicksort(4)

# for dt1 in AOT_SORT_SUPPORTED_DTYPES:
#     # cc.export(
#     print(
#         'quicksort_{}'.format(dt1),
#         'none(int64[:], int64, int64, int64[:], {}[:])'.format(dt1)
#     )
#     # )(_get_quicksort(1))

# for dt1 in AOT_SORT_SUPPORTED_DTYPES:
#     for dt2 in AOT_SORT_SUPPORTED_DTYPES:
#         # cc.export(
#         print(
#             'quicksort_{}_{}'.format(dt1, dt2),
#             'none(int64[:], int64, int64, int64[:], {}[:], {}[:])'.format(dt1, dt2)
#         )
#         # )(_get_quicksort(2))

# for dt1 in AOT_SORT_SUPPORTED_DTYPES:
#     for dt2 in AOT_SORT_SUPPORTED_DTYPES:
#         for dt3 in AOT_SORT_SUPPORTED_DTYPES:
#             # cc.export(
#             print(
#                 'quicksort_{}_{}_{}'.format(dt1, dt2, dt3),
#                 'none(int64[:], int64, int64, int64[:], {}[:], {}[:], {}[:])'.format(dt1, dt2, dt3)
#             )
#             # )(_get_quicksort(3))

# for dt1 in AOT_SORT_SUPPORTED_DTYPES:
#     for dt2 in AOT_SORT_SUPPORTED_DTYPES:
#         for dt3 in AOT_SORT_SUPPORTED_DTYPES:
#             for dt4 in AOT_SORT_SUPPORTED_DTYPES:
#                 # cc.export(
#                 print(
#                     'quicksort_{}_{}_{}_{}'.format(dt1, dt2, dt3, dt4),
#                     'none(int64[:], int64, int64, int64[:], {}[:], {}[:], {}[:], {}[:])'.format(dt1, dt2, dt3, dt4)
#                 )
#                 # )(_get_quicksort(4))

# if __name__ == "__main__":
#     cc.compile()