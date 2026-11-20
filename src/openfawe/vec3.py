"""Small 3-vector operations without numpy's per-call overhead (OF-0014, results bit for bit those of numpy).

``cross3(a, b)`` is ``numpy.cross(a, b)`` for two vectors of three components: numpy computes each component as one
product minus another (``a1*b2 - a2*b1``, ``a2*b0 - a0*b2``, ``a0*b1 - a1*b0``; every product and the difference rounded
once in IEEE double), and so does this function, on Python floats. On a 3-vector ``numpy.cross`` costs about 17 us
(broadcasting and axis handling), this about 1.6 us. Anything that is not a pair of 3-vectors goes to ``numpy.cross``.
"""
import numpy as np

_np_cross = np.cross


def cross3(a, b):
    """``numpy.cross(a, b)`` for two 3-vectors, operation for operation; other shapes are passed to numpy."""
    if type(a) is np.ndarray:
        if a.shape != (3,) or a.dtype != np.float64:
            return _np_cross(a, b)
        a = a.tolist()
    if type(b) is np.ndarray:
        if b.shape != (3,) or b.dtype != np.float64:
            return _np_cross(a, b)
        b = b.tolist()
    a0, a1, a2 = a
    b0, b1, b2 = b
    return np.array((a1*b2 - a2*b1, a2*b0 - a0*b2, a0*b1 - a1*b0))
