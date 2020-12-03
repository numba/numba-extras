def test_helloworld():
    from numba_extras.helloworld import helloworld

    msg = helloworld("world")
    assert "Hi, world" == msg
