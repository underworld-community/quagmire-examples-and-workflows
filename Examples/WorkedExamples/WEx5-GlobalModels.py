# ---
# jupyter:
#   jupytext:
#     formats: Notebooks/WorkedExamples//ipynb,Examples/WorkedExamples//py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Models on the Sphere
#
#

# +
import numpy as np
from quagmire import QuagMesh 
from quagmire import tools as meshtools
from mpi4py import MPI

import lavavu
import stripy
comm = MPI.COMM_WORLD


import matplotlib.pyplot as plt
from matplotlib import cm
# %matplotlib inline

# from scipy.ndimage.filters import gaussian_filter

# -

st0 = stripy.spherical_meshes.icosahedral_mesh(refinement_levels=7, include_face_points=True)
dm = meshtools.create_spherical_DMPlex(st0.lons, st0.lats, st0.simplices)

# +
## Read ETOPO1 data from online service
#
# Note the slicing is a mix of the data range and the integer step

import xarray

etopo_dataset = "http://thredds.socib.es/thredds/dodsC/ancillary_data/bathymetry/ETOPO1_Bed_g_gmt4.nc"
etopo_data = xarray.open_dataset(etopo_dataset)
etopo_coarse = etopo_data.sel(x=slice(-180.0,180.0,9), y=slice(-90.0,90.0,9))

lons = etopo_coarse.coords.get('x')
lats = etopo_coarse.coords.get('y')
vals = etopo_coarse['z']

# -

mesh = QuagMesh(dm, downhill_neighbours=2)

x,y = np.meshgrid(lons.data, lats.data)
height = 6.370 + 1.0e-6 * vals.data 

X,Y,Z = stripy.spherical.lonlat2xyz(np.radians(x.reshape(-1)), np.radians(y.reshape(-1)))
d,k = mesh.cKDTree.query(np.stack((X,Y,Z)).T)

# +
weights = np.zeros((mesh.npoints,))
mesh_height = np.zeros((mesh.npoints,))

d += 1.0e-10

for i in range(0, height.size):
    mesh_height[k[i]] += height.reshape(-1)[i] / d[i]
    weights[k[i]] += 1.0 / d[i]
    
mesh_height /= weights

with mesh.deform_topography():
     mesh.topography.data = mesh_height


# +
vertices = mesh.tri.points*mesh_height.reshape(-1,1)
tri = mesh.tri

lv = lavavu.Viewer(border=False, background="#FFFFFF", resolution=[600,600], near=-10.0)

# sa = lv.points("subaerial", colour="red", pointsize=0.2, opacity=0.75)
# sa.vertices(vertices[subaerial])

tris = lv.triangles("mesh",  wireframe=False, colour="#77ff88", opacity=1.0)
tris.vertices(vertices)
tris.indices(tri.simplices)
tris.values(mesh_height, label="elevation")
#tris.values(shade, label="shade")
tris.colourmap('geo', range=(6.36,6.375))
cb = tris.colourbar()

# sm = lv.points("submarine", colour="blue", pointsize=0.5, opacity=0.75)
# sm.vertices(vertices[submarine])

lv.control.Panel()
lv.control.ObjectList()
# tris.control.Checkbox(property="wireframe")
lv.control.show()
# -

height.max(), height.min()

lv.defaultcolourmaps()


