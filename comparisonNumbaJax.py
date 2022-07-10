import numpy as np
import numba as nb
import jax
import jax.numpy as jnp
import scipy
import time
from numba import prange



x = jnp.array(np.random.uniform(size=(320**3, 3)))
x = jax.device_put(x)
y = jnp.array(np.random.uniform(size=(320**3, 3)))
y = jax.device_put(y)

d = np.random.normal(size=320**3)

print("Setting interpolator")
#rbf = scipy.interpolate.RBFInterpolator(x, d, neighbors = 10)

print("Start interpolating...")
#res = rbf(y)
#print(res.shape)


def rbf(x,y):
    return jnp.exp(jnp.sum(jnp.abs(x-y)))

def func(i, tup):
    _, x,y= tup
    #res = jax.vmap(lambda y1: rbf(x, y))(x[i, :])
    return (jnp.exp(jnp.sum(jnp.abs(x[i,:]-y))), x, y)
    #return (jnp.sum(res), x,y)


def f(i, tup):
    _, x, y = tup
    res = jnp.dot(y, x[i, :])
    return (res, x, y)

f_jit = jax.jit(f)




#res = jax.vmap(lambda y1: rbf(x, y))(x[0, :])
#print(res.shape)
func_jit = jax.jit(func)


start = time.time()
res, _, _ = jax.lax.fori_loop(0, 320, func_jit, (0, x, y))
#res = jnp.dot(x, y[0, :])
#res = f_jit(x,y,0)
end = time.time()
print(end-start)
print(res.shape)

start = time.time()
#res, _, _ = jax.lax.fori_loop(0, 320*320*100, func_jit, (0, x, y))
res, _, _ = jax.lax.fori_loop(0, 320*320, func_jit, (0, x, y))
#res = jnp.dot(x, y[0, :])
#res = f_jit(x,y,0)
end = time.time()
print(end-start)
print(res.shape)


start = time.time()
#res, _, _ = jax.lax.fori_loop(0, 320**3, func_jit, (0, x, y))
#res = jnp.dot(x, y[0, :])
res = f_jit(0, (0, x, y))
end = time.time()
print(end-start)

start = time.time()
res, _, _ = jax.lax.fori_loop(0, 320, f_jit, (0, x, y))
#res = jnp.dot(x, y[0, :])
end = time.time()
print(end-start)

#start = time.time()
#res, _, _ = jax.lax.fori_loop(0, 320**3, func_jit, (0, x, y))
#end = time.time()
#print(end-start)
#print(res.shape)

#print("Starting nestesd map")
#start = time.time()
#res = jax.vmap(lambda x1: jax.vmap(lambda y1: jnp.sum(rbf(x1, y1)))(y))(x)
#end = time.time()
#print(end-start)