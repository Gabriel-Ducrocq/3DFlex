#import haiku as hk
import jax
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

from jax.config import config
config.update("jax_enable_x64", False)
def bod_fun(i, tup):
    _, x, y, indexes = tup

    return (jnp.sum(x[i]-y.at[indexes[i]]), x, y, indexes)


bod_fun_jit = jax.jit(bod_fun)
indexes = jnp.array(np.int32(np.random.uniform(0, 320, size=(320**2, 100))))
#indexes2 = jnp.array(np.int32(np.random.uniform(0, 320, size=320**3)))
x = jnp.array(np.random.normal(size=(320**3, 3)))
y = jnp.array(np.random.normal(size=(320**3, 3)))

print("Top !")
start = time.time()
res = jax.lax.fori_loop(0, 320**3, bod_fun_jit, (0, x,y, indexes))
end = time.time()
print(end-start)
print(res)

print("Top again !!")
start = time.time()
res = jax.lax.fori_loop(0, 320**3, bod_fun_jit, (0, x,y, indexes))
end = time.time()
print(end-start)
print(res)


"""
M = np.zeros(32768000)
x = np.ones(32768000)

#x = jnp.array(np.random.normal(size=3))
#y=jnp.int32(x)
#x = jnp.array(x)
#print(x == 1)
res = f(x, M)
#print(res)
"""
"""
seed = 123
key = jax.random.PRNGKey(seed)
d = jax.random.normal(key, shape=(380, 380, 380))
print("starting FFT")
test = jnp.fft.fftn(d)
print(test.shape)
"""