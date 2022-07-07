import numpy as np
import numba as nb
import jax
import jax.numpy as jnp
import scipy
from numba import prange
nb.jit(nopython=True, parallel=True)


def func(x,y):
    return jnp.sum(jnp.abs(x-y))


x = np.random.uniform(size=(320**3, 3))
y = np.random.uniform(size=(320**3, 3))
d = np.random.normal(size=320**3)

print("Setting interpolator")
rbf = scipy.interpolate.RBFInterpolator(x, d, neighbors = 10)

print("Start interpolating...")
res = rbf(y)
print(res.shape)





#res = jax.vmap(lambda y1: func(x, y))(x)
#res = jax.vmap(lambda x1: jax.vmap(lambda y1: func(x1, y1))(y))(x)

#print(res.shape)