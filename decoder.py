import haiku as hk
import pyvista as pv
import tetgen
import numpy as np
import mrcfile
import matplotlib.pyplot as plt





#encoder = hk.nets.MLP([128, 128, 128], activate_final=False, name = "encoder")

#pv.set_plot_theme('document')

#sphere = pv.Sphere()
#print(sphere.points.shape)


reader = pv.STLReader("testSTL.stl")
mesh = reader.read()
print(mesh)
tet = tetgen.TetGen(mesh)
tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
grid = tet.grid
grid.plot(show_edges=True)
point = np.ndarray(shape=(1,3), dtype=float, order='F')
elem = np.random.randint(0, tet.elem.shape[0])

print(grid)


print(tet.elem)
print(np.max(tet.elem))

print(tet.node)


print(grid.find_containing_cell(tet.node))

"""
print("Loop:")
h = []
for i in range(10000):
    print(i)
    vertices= tet.elem[elem, :]
    array_vertices = np.array([tet.node[vert] for vert in vertices])
    barycenter = np.mean(array_vertices, axis = 0)
    belong_elem = grid.find_containing_cell(barycenter)
    h.append(elem==belong_elem)

print(np.mean(h))
"""


#grid.plot(show_edges=True)
#print(sphere)
#tet = tetgen.TetGen(sphere)
#tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
#grid = tet.grid
#grid.plot(show_edges=True)


"""
with mrcfile.open('DrBphP.mrc') as mrc:
    map = mrc.header


print(map)

offset = np.array([0])
cells = np.hstack((20, np.arange(20))).astype(np.int64, copy=False)
#grid = pv.UnstructuredGrid(offset, cells, celltypes, new_map)



ax = plt.figure().add_subplot(projection='3d')
ax.voxels(new_map[100:200, 100:200, 100:200]), #facecolors=colors, edgecolor='k')
plt.show()

"""