import numpy as np
import numba as nb
import jax
import jax.numpy as jnp
import scipy
import time
from numba import prange



voxel_size = 0.82
x = jnp.array(np.random.uniform(size=(320**3, 2)))
x = jax.device_put(x)
y = jnp.array(np.random.uniform(low = 0, high=319*0.82, size=(320**3, 2)))
y = jax.device_put(y)
d = np.random.normal(size=320**3)

def first_func(j, tup):
    _, i, y, x, d, radius, voxel_size = tup
    #res =jnp.where(jnp.logical_and(jnp.logical_and(y[:, 0] < (i+1)*voxel_size + radius, y[:, 0] > (i+1)*voxel_size - radius),
    #          jnp.logical_and(y[:, 1] < (j+1)*voxel_size + radius, y[:, 1] > (j+1)*voxel_size - radius))
    #               , jnp.sum((y-x[i, :])**2), 0).sum()

    #res =jnp.where(jnp.logical_and(jnp.logical_and(y[:, 0] < (i+1)*voxel_size + radius, y[:, 0] > (i+1)*voxel_size - radius) ,
    #                jnp.logical_and(y[:, 1] < (j+1)*voxel_size + radius, y[:, 1] > (j+1)*voxel_size - radius)),
    #                               jnp.sum((y-x[i, :])**2), 0).sum()

    res = jnp.multiply(jnp.exp(jnp.sum((y - x[i, :])**2, axis = 1)), d).sum()

    #res = jnp.sum(y*resres)

    return (res, i, y, x, d, radius, voxel_size)



first_func_jit = jax.jit(first_func)

def second_func(i, tup):
    _, y, x, d, radius, voxel_size = tup
    res, _, _, _, _, _, _ = jax.lax.fori_loop(0, 320, first_func_jit, (0, i, y, x, d, radius, voxel_size))
    return (res, y, x, d, radius, voxel_size)


def unrolled(k, tup):
    _, y, x, d, radius, voxel_size = tup
    i =k//320
    j = k%320
    #res = jnp.multiply(jnp.exp(jnp.sum((y - x[i, :]) ** 2, axis=1)), d).sum()
    resres = jnp.where(jnp.logical_and(y[:, 0] < (i + 1), y[:, 0] > (i + 1)), 1, 0).sum()
    return (resres, y, x, d, radius, voxel_size)

second_func_jit = jax.jit(second_func)
unrolled_jit = jax.jit(unrolled)
#radius = 2*voxel_size
#i = 0
#res = jax.lax.fori_loop(0, 320, first_func_jit, (0, i, y, radius, voxel_size))
#print("Starting unrolled")
#start = time.time()
#res = jax.lax.fori_loop(0, 320**2, unrolled_jit, (0, y, x, d, 2*0.82, 0.82))
#end = time.time()
#print("Duration:", end -start)


print("Starting")
start = time.time()
res = jax.lax.fori_loop(0, 320, second_func_jit, (0, y, x, d, 2*0.82, 0.82))
end = time.time()
print("Duration:", end -start)


print("Starting")
start = time.time()
res = jax.lax.fori_loop(0, 320, second_func_jit, (0, y, x, d, 2*0.82, 0.82))
end = time.time()
print("Duration:", end -start)
print(res)
