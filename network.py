import haiku as hk
import jax
from jax import jit
import jax.numpy as jnp
import numpy as np
import tetgen
import jax.scipy as jscp
import pyvista as pv
import time
import scipy
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



"""
def inloop_all_W(all_u_and_variance_and_density, dictionnary):

    #We have no choice but looping over n_voxels**2, since a loop in a loop ttakes too much time to compile...
    #Compute, for each voxel x, the quantity sum_y k(x - u(y))V(y)
    #:param all_u_and_variance:
    #:param dictionnary:
    #:return:

    n_voxels = all_u_and_variance_and_density["n_voxels"]
    base_density = all_u_and_variance_and_density["base_density"]
    kernel_variance = all_u_and_variance_and_density["kernel_variance"]

    n = all_u_and_variance_and_density["n"]
    all_u_and_variance_and_density["n"] = n + 2
    vox_centroid = all_voxels_centroids.at[jnp.int32(n//n_voxels)].get()
    u = all_u.at[jnp.int32(n%n_voxels)].get()
    norm = jnp.sum(jnp.square(jnp.subtract(vox_centroid, u)))
    ker = jnp.exp(jnp.multiply(norm, -1/(2*kernel_variance)))
    #smoothed_term = ker*base_density.at[jnp.int32(n % n_voxels)].get()
    smoothed_term = base_density.at[jnp.int32(n % n_voxels)].get()
    #all_u_and_variance_and_density["W"].at[jnp.int32(n//n_voxels)].add(smoothed_term.at[0].get())
    all_u_and_variance_and_density["W"].at[jnp.int32(n//n_voxels)].set(1.0)
    return all_u_and_variance_and_density, None
"""
"""
def inloop_all_W(all_u_and_variance_and_density, voxels_centroids_and_indexes):
    #voxel_centroid = voxels_centroids_and_indexes["all_voxels_centroids"]
    #voxel_index = voxels_centroids_and_indexes["all_voxels_indexes"]
    #pix_belonging = all_u_and_variance_and_density["pix_belonging"]
    #all_u = all_u_and_variance_and_density["all_u"]
    all_u, pix_belonging = all_u_and_variance_and_density
    #t = jax.lax.slice(all_u, pix_belonging.at[0].get():pix_belonging.at[0].get()+10)
    t = all_u[jnp.arange(pix_belonging.at[0].get(), pix_belonging.at[0].get()+10)]
    #t = jnp.where(pix_belonging.at[0].get() == voxel_index.at[0].get() and pix_belonging.at[1].get() == voxel_index.at[0].get()
    #              and pix_belonging.at[2].get()  == voxel_index.at[0].get(), all_u)

    res = jnp.sum(t)
    #res = 1
    return res
    #return all_u_and_variance_and_density, res

inloop_all_W_jit = jax.jit(inloop_all_W, static_argnames=["all_u_and_variance_and_density"])
"""


def inloop_all_W(i, tup):
    _, all_u, pix_belonging, all_voxels_centroids, radius = tup
    voxel_centroid = all_voxels_centroids[i]

    #res = jnp.where(all_u[0] < radius + all_voxels_centroids[0] and all_u[0] > radius - all_voxels_centroids[0] and
    #          all_u[1] < radius + all_voxels_centroids[1] and all_u[1] > radius - all_voxels_centroids[1] and
    #          all_u[2] < radius + all_voxels_centroids[2] and all_u[2] > radius - all_voxels_centroids[2],
    #          jnp.exp(jnp.sum((all_u - voxel_centroid)**2)), 0).sum()

    #belong = pix_belonging[i-1:i+1]
    #res = jnp.sum(belong)
    t = jnp.multiply(all_u[:, 0], voxel_centroid[0])
    #res1 = jnp.where(all_u[:, 0] < -660, jnp.sum(all_u - voxel_centroid, axis = 1), 0)
    #res = jnp.sum(res1)
    #res = jnp.where(all_u.at[:, 0].get() < 1 and all_u.at[:, 0].get()>0.7, all_u, 0).sum()
    #d = all_u[all_u[0] < 0]
    #t = all_u[pix_belonging[0][0], pix_belonging[0][1]]
    res = jnp.sum(t)
    return (res, all_u, pix_belonging, all_voxels_centroids, radius)

inloop_all_W_jit = jax.jit(inloop_all_W)


import numba as nb
from numba import prange
nb.jit(nopython=True, parallel=True)
def test(all_u, voxels_centroids):
    res = np.zeros(320**3)
    for i in prange(320**3):
        res[i] = np.sum(np.dot(all_u[0,:],voxels_centroids[i, 0]))

    return res

all_u = np.random.normal(size=(320**3, 3))
all_voxels_centroids = np.random.normal(size=(320**3, 3))

print("Launching numba func")
start = time.time()
res = test(all_u, all_voxels_centroids)
end = time.time()
print("Duration:", end - start)
print(res)


print("Launching numba func")
start = time.time()
res = test(all_u, all_voxels_centroids)
end = time.time()
print("Duration:", end - start)
print(res)

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


class Compute_all_W():
    def __init__(self, kernel_variance, all_voxels_centroids, name="all_W"):
        #super().__init__(name=name)
        self.kernel_variance = jnp.float32(kernel_variance)
        self.all_voxels_centroids = jnp.array(all_voxels_centroids)
        self.n_voxels = all_voxels_centroids.shape[0]
        self.voxel_size = 0.82
        self.voxels_indexes = jnp.array(np.int32(np.random.uniform(0, 320, size=(320**3, 3))))

    def __call__(self, x):
        """
        base_density, all_u = x["base_density"], x["all_u"]
        pix_belonging = jnp.array(jnp.int32(jnp.divide(all_u, self.voxel_size)))
        #x.update({"all_u":all_u, "kernel_variance":self.kernel_variance, "pix_belonging":pix_belonging,
        #          "voxels_indexes":self.voxels_indexes})
        x = (pix_belonging, self.voxels_indexes)
        all_voxels_centroids_and_indexes = {"all_voxels_centroids":self.all_voxels_centroids,
                                            "all_voxels_indexes":pix_belonging}
        _, res = jax.lax.scan(inloop_all_W_jit, x, all_voxels_centroids_and_indexes)
        #all_u_transp = jnp.transpose(all_u)
        #print(all_voxels_centroids.shape)
        #print(all_u_transp.shape)
        #res = jnp.dot(all_voxels_centroids.at[0].get(), jnp.transpose(all_u))
        """
        base_density, all_u = x["base_density"], x["all_u"]
        #pix_belonging = tuple(map(tuple, (np.int32(np.divide(all_u, self.voxel_size)))))
        #pix_belonging = jnp.array(np.int32(np.divide(all_u, self.voxel_size)))
        #for i in range(self.n_voxels):
        #    res = inloop_all_W_jit(all_u, pix_belonging)

        all_u = np.random.normal(size=(320**3, 3))
        all_voxels_centroids = np.random.normal(size=(320**3, 3))
        print("Launching loop")
        #res = jax.lax.fori_loop(0, 320**3, inloop_all_W_jit, (0, all_u, pix_belonging, self.all_voxels_centroids, 2*0.82))
        res = test(all_u, all_voxels_centroids)
        return res




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
box_size_x = box_size_y = box_size_z = 262.4
voxel_sizes = (box_size_x/n_voxels[0], box_size_y/n_voxels[1], box_size_z/n_voxels[2])
seed = 123
key = jax.random.PRNGKey(seed)
base_density = jax.random.normal(key, shape=(jnp.product(jnp.array(n_voxels)), 1) )
d = np.load("data/DrBphP/data.npy", allow_pickle=True)
d = d.item()
all_inv_matrices = d["all_inv_matrices"]
voxels_elements = d["voxels_elements"]
all_voxels_centroids = jnp.array(d["all_voxels_centroids"])



convection_vector = jax.random.normal(key, shape=(tet.elem.shape[0], 3))
all_coeffs = jnp.ones((tet.elem.shape[0], 4, 3))
all_u = jnp.ones((320*320*320, 3))


def _compute_all_A_and_B_matrices(convection_vectors):
    net = Compute_all_A_and_b_matrices(tet.elem, all_inv_matrices)
    return net(convection_vectors)

def _compute_all_u(all_coeffs):
    net = Compute_all_u(voxels_elements, all_voxels_centroids)
    return net(all_coeffs)

def _compute_all_W(x):
    net = Compute_all_W(10000.0, all_voxels_centroids)
    return net(x)



compute_all_A_and_B_matrices = hk.without_apply_rng(hk.transform(_compute_all_A_and_B_matrices))
compute_all_u =hk.without_apply_rng(hk.transform(_compute_all_u))
#compute_all_W =hk.without_apply_rng(hk.transform(_compute_all_W))

params = compute_all_A_and_B_matrices.init(key, convection_vector)
params = compute_all_u.init(key, all_coeffs)
print("Initializing W")
#compute_all_W_init_jit = jax.jit(compute_all_W.init)
#compute_all_W_apply_jit = jax.jit(compute_all_W.apply)
#params =compute_all_W_init_jit(key, {"all_u":all_u, "base_density":base_density})
print("Done initializing")



test =compute_all_u.apply(all_coeffs=compute_all_A_and_B_matrices.apply(
                          convection_vectors=convection_vector, params=params), params=params)



print(test.shape)
print("Launching heavy")
start = time.time()
res = _compute_all_W(x={"all_u":test, "base_density":base_density})
print("Duration:", time.time()-start)




#new_input = {"all_u":test, "base_density":base_density}
#print("Lauching heavy")
#res = compute_all_W_apply_jit(x=new_input, params=params)

#print(res)
#base_density = base_density.reshape((320,320, 320))
#print(base_density.shape)
#start = time.time()
#test_interpol = jscp.ndimage.map_coordinates(base_density, test[:,:].T, order=1)
#print(test_interpol.shape)
#test_interpol = scipy.ndimage.map_coordinates(base_density, test[:,:].T, order=3)
#end = time.time()
#print(end-start)



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

