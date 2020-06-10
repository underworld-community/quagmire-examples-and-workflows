# ---
# jupyter:
#   jupytext:
#     formats: Notebooks/Tutorial//ipynb,Examples/Tutorial//py:light
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

# ## quagmire.mesh MeshVariable
#
# Like Underworld, quagmire provides the concept of a "variable" which is associated with a mesh. These are parallel data structures on distributed meshes that support various capabilities such as interpolation, gradients, save and load, as well as supporting a number of mathematical operators through the `quagmire.function` interface (examples in the next notebook). 
#
#

from quagmire.tools import meshtools
from quagmire import QuagMesh
from quagmire.mesh import MeshVariable
import numpy as np  

# ### Working mesh
#
# First we create a basic mesh so that we can define mesh variables and obtain gradients etc.

# +
minX, maxX = -5.0, 5.0
minY, maxY = -5.0, 5.0,
dx, dy = 0.02, 0.02

x,y, bound = meshtools.generate_elliptical_points(minX, maxX, minY, maxY, dx, dy, 60000, 300)
DM = meshtools.create_DMPlex_from_points(x, y, bmask=bound)
mesh = QuagMesh(DM, downhill_neighbours=1)
# -

# ### Basic usage
#
# Mesh variables can be instantiated directly or by adding a new variable to an existing mesh. 
#

# +
phi = mesh.add_variable(name="PHI(X,Y)")
psi = mesh.add_variable(name="PSI(X,Y)")

# is equivalent to

phi1 = MeshVariable(name="PSI(X,Y)", mesh=mesh)
psi1 = MeshVariable(name="PHI(X,Y)", mesh=mesh)
# -

# Mesh variables store their data in a PETSc distributed vector with values on the local mesh accessible through a numpy interface (via to petsc4py). For consistency with `underworld`, the numpy array is accessed as the `data` property on the variable as follows

phi.data = np.sin(mesh.coords[:,0])**2.0 
psi.data = np.cos(mesh.coords[:,0])**2.0 * np.sin(mesh.coords[:,1])**2.0 

# Note that the following is not allowed
#
# ```python
# phi.data[0] = 1.0
# ```
#
# and nor is any other change to a single value in the array. This is done so that we can be sure that
# the values in the array are always synchronised across processors at the end of an assignment. It is also
# done to control cases where there are dependencies on the variable that go beyond synchronisation (for example,
# changing the topography variable rebuilds the flow pathways in a surface process context). 
#
# You can work with a local copy of the vector and update all at once if you need to build incremental changes to values, work without synchronisation across processors or avoid rebuilding of dependent quantities. 

# A MeshVariable object responds to a `print` statement by stating what it is and its name. To print the contents of the variable (locally), access the values through the data property:

print(phi, "|", psi)
print(phi.data)

# Mesh variables can be read only (locked). The RO (read only) and RW (read / write) markers are shown when the variable is printed. 
#
# ```python
# phi.lock()
# print(phi)
# phi.unlock()
# print(phi)
# ```
#
# Generally locking is done to prevent changes to a variable's data because additional updates depend on controlling when changes are made. Access to `lock` and `unlock` is 

phi.lock()
print(phi)
phi.unlock()
print(phi)

# ### Parallel support
#
# The `MeshVariable` class has a `sync` method that, when called, will replace shadow information with values from adjacent sections of the decomposition (or optionally, merge values in the shadow zone - an operation that should be used with caution for global reduction type operations). 
#
# If you alter data in the shadow zone in a way that cannot be guaranteed to be the same on another processor, then some form of synchronisation is required when you are done. This is not fully automated as there may be several stages to your updates that you only want to synchronise at the end. But, still, be careful !
#

# +
phi.sync()

phi.sync(mergeShadow=True) # this will add the values from each processor in parallel
# -

# These kinds of parallel operations must be called on every processor or they will block while waiting for everyone to finish. Be careful not to call sync inside a conditional which may be executed differently 
#
# ```python
#
# import quagmire
#
# # Don't do this (obviously)
# if quagmire.rank == 0:
#     phi.sync()   
#    
# # or something a little bit less obvious
# if delta_phi > 0.0:
#     phi.sync()
#     
# # This might be OK but it is not required
# if quagmire.size > 1:
#     phi.sync()
#
# ```

# ### Evaluate method and fn_gradient
#
# MeshVariables support the `evaluate` method (because they are `quagmire.functions`) which is useful as it generalises various interfaces that are available to access the data. If a mesh is supplied, then evaluate checks to see if this corresponds to the mesh associated with the mesh variable and returns the raw data if it does. Otherwise the mesh coordinates are used for interpolation. If two coordinate arrays are supplied then these are passed to the interpolator. 
#
# NOTE: the interpolator will also extrapolate and may (is likely to) produce poor results for off-processor coordinates. If this is a problem, the `MeshVariable.interpolate` method can be accessed directly with the `err` optional argument. 
#

# +
## Raw nodal point data for the local domain

print(phi.data)
print(phi.evaluate(mesh))
print(phi.evaluate(phi._mesh)) 

## interpolation at a point 

print(phi.interpolate(0.01,1.0))
print(phi.evaluate(0.01, 1.0))
# -


# Mesh based variables can be differentiated in (X,Y). There is a `gradient` method that supplies the coefficients of the derivative surface at the nodal points (these may then need to be interpolated). A more general interface is also provided in the form of a function which can be evaluated (as above):
#

# +
dpsidx_nodes, dpsidy_nodes = psi.gradient()
print(dpsidx_nodes)
print(dpsidy_nodes)

dpsidx_fn = psi.fn_gradient[0] # (0) for X derivative, (1) for Y
print(dpsidx_fn.evaluate(mesh))
print(dpsidx_fn.evaluate(0.01, 1.0))

dpsidx_fn
# -

# ### Visualisation

# The following should all evaluate to zero everywhere and so act as a test on the accuracy of the gradient operator 

# +
import lavavu

xyz = np.column_stack([mesh.tri.points, np.zeros_like(phi.data)])

lv = lavavu.Viewer(border=False, background="#FFFFFF", resolution=[1000,600], near=-10.0)

tris = lv.triangles("triangles",  wireframe=False, colour="#77ff88", opacity=1.0)
tris.vertices(xyz)
tris.indices(mesh.tri.simplices)
tris.values(phi.evaluate(mesh), label="phi")
tris.values(psi.evaluate(mesh), label="psi")
tris.values(dpsidx_nodes, label="dpsidx_nodes")



tris.colourmap("elevation")
cb = tris.colourbar()

lv.control.Panel()
lv.control.Range('specular', range=(0,1), step=0.1, value=0.4)
lv.control.Checkbox(property='axis')
lv.control.ObjectList()
tris.control.Checkbox(property="wireframe")
tris.control.List(options = ["phi", "psi", "dpsidx_nodes"], property="colourby", value="psi", command="redraw", label="Display:")
lv.control.show()
# -


