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


class Network():
    def __init__(self, base_density, mesh_elements, mesh_nodes, voxels_sizes, n_voxels, unstructured_grid, dim_latent,
                 kernel_variance = 1):
        self.dim_latent = dim_latent
        self.kernel_variance = kernel_variance
        self.base_density= base_density
        self.mesh_elements = mesh_elements
        self.n_elements = mesh_elements.shape[0]
        self.mesh_nodes = mesh_nodes
        self.n_nodes = mesh_nodes.shape[0]
        self.voxel_sizes = voxels_sizes
        self.n_voxels = n_voxels
        self.grid = unstructured_grid
        self.num_pixels = len(base_density)

        print("Creating centroids:")
        #self.create_pixel_centroids(n_voxels, voxels_sizes)
        print("Getting pixel elements")
        #self.get_pixels_element()


        print("Declaring")
        self.compute_inverse_matrix_jit = jax.jit(self.compute_inverse_matrix)

        #print("Computing inverse matrices")
        #start = time.time()
        #self.all_inv_matrices = self.compute_all_inverse_matrices()
        #self.all_inv_matrices.block_until_ready()
        #print(time.time()-start)

        #self.encoder = hk.nets.MLP([dim_latent, 128, 128, self.n_nodes * 3], activate_final=False, name="encoder")

    def create_pixel_centroids(self, n_voxels, voxel_sizes):
        self.centroids_x = np.arange(0, n_voxels[0])*voxel_sizes[0] + voxel_sizes[0]/2
        self.centroids_y = np.arange(0, n_voxels[1])*voxel_sizes[1] + voxel_sizes[1]/2
        self.centroids_z = np.arange(0, n_voxels[2])*voxel_sizes[2] + voxel_sizes[2]/2

    def get_pixels_element(self):
        print("Computing centroids")
        all_pixels_centroids = np.array(list(itertools.product(self.centroids_x, self.centroids_y, self.centroids_z)))
        print("Getting elements")
        pixels_element = self.grid.find_containing_cell(all_pixels_centroids)
        print("Setting centroids to jax")
        self.all_pixels_centroids = jnp.array(all_pixels_centroids)
        print("Setting pixels elements to jax")
        self.pixels_element = jnp.array(pixels_element)

    def compute_inverse_matrix(self, vert1, vert2, vert3, vert4):
        m = jnp.concatenate([jnp.ones((4,1)),jnp.array([ vert1, vert2, vert3, vert4])], axis = 1)
        return jnp.linalg.pinv(m)

    def compute_all_inverse_matrices(self):
        all_inv_matrices = np.zeros((self.n_elements, 4, 4))
        for i in range(self.n_elements):
            elt1, elt2, elt3, elt4 = self.mesh_elements[i]
            all_inv_matrices[i, :, :] = self.compute_inverse_matrix_jit(self.mesh_nodes[elt1], self.mesh_nodes[elt2],
                                                                          self.mesh_nodes[elt3], self.mesh_nodes[elt4])

        return all_inv_matrices


    def forward_encoder(self, z):
        convection_vectors = self.encoder(z)
        convection_vectors = jnp.reshape(convection_vectors, (-1, 3))
        return convection_vectors


    def compute_all_A_and_b_matrices(self, convection_vectors):
        """
        Computes the matrices A_j and vector B_j for every element j of the FEM.
        :param convection_vectors: array of size (n_nodes, 3) of all the convection vectors at tthe FEM vertices
        :return: array(n_elements,
        """
        all_coeffs = jnp.zeros((self.n_elements, 4, 3))
        all_elts = jnp.arange(0, self.n_elements)
        for elt in all_elts:
            vertices_numbers = self.mesh_elements[elt]
            convection_vectors_elt = convection_vectors[vertices_numbers]
            inv_mat = self.all_inv_matrices[elt, :, :]
            coeffs_elt = jnp.dot(inv_mat, convection_vectors_elt)
            all_coeffs = all_coeffs.at[elt].set(coeffs_elt)

        return all_coeffs

    def compute_u(self, pix, all_coeffs):
        """

        :param coordinates_centroid: coordinate of the centroid to perform u(coordinates_centroid)
        :param elt: integer, elements to which the voxel belongs to.
        :param all_coeffs: array (n_elements, 4, 3) coefficients of the matrices A and bias b of interpolation.
        :return: array (1, 3), the value u(coordinates_centroid).
        """
        elt = self.pixels_element[pix]
        coeffs = jnp.concatenate([jnp.ones(1), all_coeffs[elt]])
        return jnp.dot(jnp.concatenate([jnp.ones(1),self.all_pixels_centroids[pix]]), coeffs)

    def compute_all_u(self, all_coeffs):
        """

        :param all_coeffs: array (n_elements, 4, 3) coefficients of the matrices A and bias b of interpolation.
        :return: array (N_voxels, 3), the value u(coordinates_centroid) for each voxel.
        """
        all_u = jnp.zeros((self.num_pixels, 3))
        all_voxels = jnp.arange(0, self.num_pixels)
        for vox in all_voxels:
            u_at_vox = self.compute_u(vox, all_coeffs)
            all_u.at[vox].set(u_at_vox)

        return all_u

    def compute_W_pixel(self, voxel, all_u):
        """

        :param voxel: voxel number
        :param all_u: array (N_voxel, 3), the new - transformed - coordinates
        :return: interpolation with exponential kernel, at the specified voxel.
        """
        return jnp.exp(jnp.divide(jnp.multiply(-1, jnp.sum(jnp.square(jnp.substract(self.all_pixels_centroids[voxel] - all_u)), axis = 1))
                                  , 2*self.kernel_variance))


    def compute_W(self, all_u):
        """

        :param all_u: array (N_voxels, 3), the new - transformed - coordinates
        :return: array (N_voxels,) the values of the convected density W at each voxel.
        """
        all_W = jnp.zeros((self.num_pixels, 3))
        all_voxels = jnp.arange(0, self.num_pixels)
        for vox in all_voxels:
            w_at_vox = self.compute_W_pixel(vox, all_u)
            all_W.at[vox].set(w_at_vox)

        return all_W

    #def pipeline(self, base_density, latent variables, ):



        

reader = pv.STLReader("testSTL.stl")
mesh = reader.read()
tet = tetgen.TetGen(mesh)
tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
grid = tet.grid

with mrcfile.open('DrBphP.mrc') as mrc:
    map = mrc.header


n_voxels = (320, 320, 320)
box_size_x = box_size_y = box_size_z = 262.4
voxel_sizes = (box_size_x/n_voxels[0], box_size_y/n_voxels[1], box_size_z/n_voxels[2])


start = time.time()
seed = 123
key = jax.random.PRNGKey(seed)
base_density = jax.random.normal(key, shape=(jnp.product(jnp.array(n_voxels)),) )
net = Network(base_density, tet.elem, tet.node, voxel_sizes, n_voxels, grid, 5)
