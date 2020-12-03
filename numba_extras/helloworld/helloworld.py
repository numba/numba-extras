from numba import njit


@njit
def helloworld(msg):
    return "Hi, " + msg
