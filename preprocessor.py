import haiku as hk
import numpy as np
import jax


def computing_centroid_pixels(n_voxels, voxel_sizes):
    centroids_x = np.arange(0, n_voxels[0])*voxel_sizes[0] + voxel_sizes[0]/2
    centroids_y = np.arange(0, n_voxels[1])*voxel_sizes[1] + voxel_sizes[1]/2
    centroids_z = np.arange(0, n_voxels[2])*voxel_sizes[2] + voxel_sizes[2]/2
    all_pixels_centroids = np.array(list(itertools.product(centroids_x, centroids_y, centroids_z)))
    return all_pixels_centroids


def getting_pixels_elements(all_pixels_centroids, grid):
    return grid.find_containing_cell(all_pixels_centroids)


def compute_inverse_matrix(vert1, vert2, vert3, vert4):
    m = jnp.concatenate([jnp.ones((4, 1)), jnp.array([vert1, vert2, vert3, vert4])], axis=1)
    return jnp.linalg.pinv(m)

compute_inverse_matrix_jit = jax.jit(self.compute_inverse_matrix)


def compute_all_inverse_matrices():
    all_inv_matrices = jnp.zeros((self.n_elements, 4, 4))
    for i in range(self.n_elements):
        elt1, elt2, elt3, elt4 = self.mesh_elements[i]
        all_inv_matrices[i, :, :] = self.compute_inverse_matrix_jit(self.mesh_nodes[elt1], self.mesh_nodes[elt2],
                                                                    self.mesh_nodes[elt3], self.mesh_nodes[elt4])

    return all_inv_matrices