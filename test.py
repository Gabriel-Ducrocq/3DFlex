import haiku as hk
import jax
from jax import jit
import jax.numpy as jnp
import time

class Test():
    def __init__(self, n):
        self.n = n
        self.test_jitted = jax.jit(self.test)

    def test(self, x):
        for i in range(self.n):
            x = jnp.add(x, 1)

        return x


test = Test(100000)
#test.test_jitted(10)
ar = jnp.array([[1, 2, 3], [4, 5, 6]])
print(ar)
"""
start = time.time()
a = test.test_jitted(10)
end = time.time()
print(end-start)
start = time.time()
b = test.test(10)
end = time.time()
print(end-start)
print(a)
print(b)
"""