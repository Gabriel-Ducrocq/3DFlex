import haiku as hk
import jax
from jax import jit
import jax.numpy as jnp
import time
import numpy as np
from numba import jit
from numba import prange


"""
def test(elt, r):
    elt["elt"] +=1
    return elt, None


fin, res = jax.lax.scan(test, {"elt":0, "B":1}, None, length=32768000)
print(res)
print(fin)


@jit(nopython=True, parallel=False)
def f(a, x, M):
    for i in prange(32768000):
        for j in prange(32768000):
            elt1 = x[i]
            r = elt1[0]**2 + elt1[1]**2 + elt1[2]**2
            a += r


    return a



M = np.random.normal(size=(32768000, 3))
x = np.random.normal(size=(32768000, 3))

res = f(0, x, M)
print(res)
"""

seed = 123
key = jax.random.PRNGKey(seed)
d = jax.random.normal(key, shape=(380, 380, 380))
print("starting FFT")
test = jnp.fft.fftn(d)
print(test.shape)