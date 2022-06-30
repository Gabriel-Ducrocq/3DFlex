import haiku as hk
import jax
from jax import jit
import jax.numpy as jnp
import time
import numpy as np



def test(elt, r):

    return elt, r["A"]+elt["test"]


fin, res = jax.lax.scan(test, {"elt":0, "test":1}, {"A":np.arange(0, 10), "B":np.arange(10, 20)})
print(res)
print(fin)