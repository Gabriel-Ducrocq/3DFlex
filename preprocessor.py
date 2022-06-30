import haiku as hk
import numpy as np
import jax
import jax.numpy as jnp
import itertools


def computing_centroid_voxels(n_voxels, voxel_sizes):
    centroids_x = np.arange(0, n_voxels[0])*voxel_sizes[0] + voxel_sizes[0]/2
    centroids_y = np.arange(0, n_voxels[1])*voxel_sizes[1] + voxel_sizes[1]/2
    centroids_z = np.arange(0, n_voxels[2])*voxel_sizes[2] + voxel_sizes[2]/2
    all_voxels_centroids = np.array(list(itertools.product(centroids_x, centroids_y, centroids_z)))
    return all_voxels_centroids


def getting_voxels_elements(all_voxels_centroids, grid):
    return grid.find_containing_cell(all_voxels_centroids)


def compute_inverse_matrix(vert1, vert2, vert3, vert4):
    m = jnp.concatenate([jnp.ones((4, 1)), jnp.array([vert1, vert2, vert3, vert4])], axis=1)
    return jnp.linalg.pinv(m)

compute_inverse_matrix_jit = jax.jit(compute_inverse_matrix)


def compute_all_inverse_matrices(mesh_elements, mesh_nodes):
    n_elements = mesh_elements.shape[0]
    all_inv_matrices = np.zeros((n_elements, 4, 4))
    for i in range(n_elements):
        elt1, elt2, elt3, elt4 = mesh_elements[i]
        all_inv_matrices[i, :, :] = compute_inverse_matrix_jit(mesh_nodes[elt1], mesh_nodes[elt2],
                                                                    mesh_nodes[elt3], mesh_nodes[elt4])

    return all_inv_matrices



def preprocessing_pipeline(grid, mesh_elements, mesh_nodes, n_voxels, voxel_sizes):
    all_voxels_centroids = computing_centroid_voxels(n_voxels, voxel_sizes)
    voxels_elements = getting_voxels_elements(all_voxels_centroids, grid)
    all_inv_matrices = compute_all_inverse_matrices(mesh_elements, mesh_nodes)
    return all_voxels_centroids, voxels_elements, all_inv_matrices


