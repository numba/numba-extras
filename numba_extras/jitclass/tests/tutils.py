import re
import pytest
from contextlib import AbstractContextManager


class raises_with_msg(AbstractContextManager):
    def __init__(self, exc_type, msg):
        if not isinstance(msg, (str, re.Pattern)):
            raise RuntimeError(f"msg must be string or Patterh. Got {type(msg)}")

        self.exc_type = exc_type
        self.msg = msg

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            pytest.fail("Exception wasn't raised")

        if self.exc_type != exc_type:
            return False

        exc_msg = str(exc_value)
        if isinstance(self.msg, str):
            descr = f"Expected message '{self.msg}' and actual error message '{exc_msg} doesn't match'"
            assert self.msg == exc_msg, descr
        else:
            assert isinstance(self.msg, re.Pattern)
            descr = f"Expected pattern '{self.msg.pattern}' doesn't match actual error '{exc_msg}'"
            assert self.msg.match(exc_msg) is not None, descr

        return True
