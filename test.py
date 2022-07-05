#import haiku as hk
#import jax
#from jax import jit
import jax.numpy as jnp
import time
import numpy as np
from numba import jit
from numba import prange



def test(elt, r):
    elt["elt"] +=1
    return elt, None


#fin, res = jax.lax.scan(test, {"elt":0, "B":1}, None, length=32768000)
#print(res)
#print(fin)


@jit(nopython=True, parallel=False)
def f(x, M):
    for i in prange(320**6):
    #for j in prange(32768000):
        j = i%320**3
        elt1 = x[j]
        M[j] = elt1

    return M


M = np.zeros(32768000)
x = np.ones(32768000)

x = jnp.array(np.random.normal(size=3))
y=jnp.int32(x)
x = jnp.array(x)
print(x == 1)
#res = f(x, M)
#print(res)
"""
seed = 123
key = jax.random.PRNGKey(seed)
d = jax.random.normal(key, shape=(380, 380, 380))
print("starting FFT")
test = jnp.fft.fftn(d)
print(test.shape)
"""