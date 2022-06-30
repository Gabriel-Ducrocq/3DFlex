import haiku as hk
import jax
from jax import jit
import jax.numpy as jnp
import numpy as np
import tetgen
import pyvista as pv
import time
import mrcfile
import itertools
import preprocessor



def inloop_A_and_b(convection_vectors, dictionnary):
    vertices_numbers = dictionnary["mesh_elements"]
    convection_vectors_elt = convection_vectors.at[vertices_numbers].get()
    inv_mat = dictionnary["all_inv_matrices"]
    return convection_vectors, inv_mat @ convection_vectors_elt

inloop_A_and_b_jit = jax.jit(inloop_A_and_b)




def inloop_all_u(all_coeffs, dictionnary):
    """

    :param all_coeffs: array (n_elements, 4, 3), array of interpolation coefficients to compute the displacement u, v, w for each element
    :param dictionnary:
    :return: all_coeffs, displacement vector (u,v,w) expressed at the voxel centroid
    """
    voxels_elements = dictionnary["voxels_elements"]
    all_voxels_centroids = dictionnary["all_voxels_centroids"]
    return all_coeffs, jnp.dot(jnp.concatenate([jnp.ones(1), all_voxels_centroids]), all_coeffs[voxels_elements])

inloop_all_u_jit = jax.jit(inloop_all_u)




def inloop_all_W(all_u_and_variance_and_density, dictionnary):
    """
    Compute, for each voxel x, the quantity ssum_y k(x - u(y))V(y)
    :param all_u_and_variance:
    :param dictionnary:
    :return:
    """
    #kernel_variance = all_u_and_variance_and_density["kernel_variance"]
    #all_voxels_centroids = dictionnary["all_voxels_centroids"]
    #all_u = all_u_and_variance_and_density["all_u"]
    base_density = all_u_and_variance_and_density["base_density"]
    return all_u_and_variance_and_density, jnp.multiply(base_density, base_density)
    #return all_u_and_variance_and_density, jnp.sum(jnp.multiply(jnp.exp(
    #    jnp.divide(jnp.multiply(-1, jnp.sum(jnp.square(jnp.subtract(all_voxels_centroids, all_u)), axis=1))
    #               , 2 * kernel_variance)), base_density))

inloop_all_W_jit = jax.jit(inloop_all_W)


class Compute_all_A_and_b_matrices(hk.Module):
    def __init__(self, mesh_elements, inv_matrices, name="A_and_b"):
        super().__init__(name=name)
        self.mesh_elements = jnp.array(mesh_elements)
        self.n_elements = mesh_elements.shape[0]
        self.inv_matrices = jnp.array(inv_matrices)

    def __call__(self, convection_vectors):
        """
        Computes the matrices A_j and vector B_j for every element j of the FEM.
        :param convection_vectors: array of size (n_nodes, 3) of all the convection vectors at tthe FEM vertices
        :return: array(n_elements,
        """
        mesh_elements = self.mesh_elements
        all_inv_matrices = self.inv_matrices
        xs = {"mesh_elements": jnp.array(mesh_elements), "all_inv_matrices": all_inv_matrices}
        convection_vectors_again, all_A_and_b_matrices = jax.lax.scan(inloop_A_and_b_jit, convection_vectors, xs)
        return all_A_and_b_matrices



class Compute_all_u(hk.Module):
    def __init__(self, voxels_elements, all_voxels_centroids, name="all_u"):
        super().__init__(name=name)
        self.voxels_elements = jnp.array(voxels_elements)
        self.all_voxels_centroids = jnp.array(all_voxels_centroids)
        self.num_voxels = voxels_elements.shape[0]

    def __call__(self, all_coeffs):
        """

        :param all_coeffs: array (n_elements, 4, 3) coefficients of the matrices A and bias b of interpolation.
        :return: array (N_voxels, 3), the value u(coordinates_centroid) for each voxel.
        """
        xs = {"voxels_elements":self.voxels_elements, "all_voxels_centroids":self.all_voxels_centroids}
        _, all_u = jax.lax.scan(inloop_all_u_jit, all_coeffs, xs)
        return all_u


class Compute_all_W(hk.Module):
    def __init__(self, kernel_variance, all_voxels_centroids, name="all_W"):
        super().__init__(name=name)
        self.kernel_variance = kernel_variance
        self.all_voxels_centroids = all_voxels_centroids

    def __call__(self, x):
        base_density, all_u = x["base_density"], x["all_u"]
        xs = {"all_voxels_centroids":self.all_voxels_centroids}
        x.update({"all_u":all_u, "kernel_variance":self.kernel_variance})
        print("Launching jax loop")
        _, all_W = jax.lax.scan(inloop_all_W_jit, x, xs)
        return all_W




reader = pv.STLReader("testSTL.stl")
mesh = reader.read()
print("Done reading")
tet = tetgen.TetGen(mesh)
tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
print("Done tetra")
grid = tet.grid
print("Done grid")
"""
with mrcfile.open('DrBphP.mrc') as mrc:
    map = mrc.header


n_voxels = (320, 320, 320)
box_size_x = box_size_y = box_size_z = 262.4
voxel_sizes = (box_size_x/n_voxels[0], box_size_y/n_voxels[1], box_size_z/n_voxels[2])


start = time.time()
seed = 123
key = jax.random.PRNGKey(seed)
base_density = jax.random.normal(key, shape=(jnp.product(jnp.array(n_voxels)),) )
#net = Network(base_density, tet.elem, tet.node, voxel_sizes, n_voxels, grid, 5)
all_voxels_centroids, voxels_element, all_inv_matrices = preprocessor.preprocessing_pipeline(grid, tet.elem, tet.node, n_voxels, voxel_sizes)
data = {"all_voxels_centroids":all_voxels_centroids, "voxels_elements":voxels_element, "all_inv_matrices":all_inv_matrices}
np.save("data/DrBphP/data.npy", data, allow_pickle=True)
"""
n_voxels = (320, 320, 320)
seed = 123
key = jax.random.PRNGKey(seed)
base_density = jax.random.normal(key, shape=(jnp.product(jnp.array(n_voxels)),) )
d = np.load("data/DrBphP/data.npy", allow_pickle=True)
d = d.item()
all_inv_matrices = d["all_inv_matrices"]
voxels_elements = d["voxels_elements"]
all_voxels_centroids = d["all_voxels_centroids"]



convection_vector = jnp.ones((tet.elem.shape[0], 3))
all_coeffs = jnp.ones((tet.elem.shape[0], 4, 3))


def _compute_all_A_and_B_matrices(convection_vectors):
    net = Compute_all_A_and_b_matrices(tet.elem, all_inv_matrices)
    return net(convection_vectors)

def _compute_all_u(all_coeffs):
    net = Compute_all_u(voxels_elements, all_voxels_centroids)
    return net(all_coeffs)

def _compute_all_W(x):
    net = Compute_all_W(1, all_voxels_centroids)
    return net(x)



compute_all_A_and_B_matrices = hk.without_apply_rng(hk.transform(_compute_all_A_and_B_matrices))
compute_all_u =hk.without_apply_rng(hk.transform(_compute_all_u))
compute_all_W =hk.without_apply_rng(hk.transform(_compute_all_W))


params = compute_all_A_and_B_matrices.init(key, convection_vector)
params = compute_all_u.init(key, all_coeffs)



test =compute_all_u.apply(all_coeffs=compute_all_A_and_B_matrices.apply(
                          convection_vectors=convection_vector, params=params), params=params)

print("Done computing test")
test = compute_all_W.apply(x={"base_density":base_density,
                      "all_u":test}, params = params)
print(params)
print(test.shape)
print(all_voxels_centroids.shape)


#def _compute_all_u(all_coeffs):
#    return net.compute_all_u(all_coeffs)

#compute_all_u = hk.without_apply_rng(hk.transform(_compute_all_u))

#def _compute_all_W(all_u):
#    return net.compute_all_W(all_u)

#compute_all_W = hk.without_apply_rng(hk.transform(_compute_all_W))

"""
print("Generating convection vector")
#convection_vector = jax.random.normal(key, shape = (tet.elem.shape[0], 3))
convection_vector = jnp.ones((tet.elem.shape[0], 3))
print("Initializing params:")
jax_init = jax.jit(compute_all_A_and_B_matrices.init)
params = jax_init(key, convection_vector)
print("params:")
print(params)
all_A_b_matrices = compute_all_A_and_B_matrices.apply(x=convection_vector)
print(all_A_b_matrices.shape)
"""

