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

# ## Quagmire.function
#
# Like Underworld, quagmire provides a function interface that can be used to compose data and operations on the fly in order to construct model equations independent of whatever approach is used for solution. 
#
# Noteably, these are _lazy_ function that are only evaluated when needed. More importantly, when evaluated, they also use the current state of any variables in their definition and so can be placed within timestepping loops where they will always use the information of the current timestep.
#
# There are three kinds of lazy functions available in quagmire:
#
#   - `MeshVariable` data containers that hold information on the mesh and can return that information at any point by interpolation (or, less reliably by extrapolation) and can also provide the gradient of the data using a cubic spline interpolant (see the documentation for `stripy` for details).
#   
#   - `parameter` is a floating point value that can be used for coefficients in an equation. The value of the parameter can be updated.
#   
#   - `virtual` variables which are operations on `MeshVariables` and `parameters` and contain no data record. 
#   
#   
# These lazy functions are members of the `LazyEvaluation` class that defines the following methods / data
#
#   - `evaluate(mesh | X, Y)` computes a snapshot of the value(s) at the mesh points of `mesh` or at the points given by X and Y
#   
#   - `fn_gradient(dir)` is a lazy function that can be evaluated to obtain the gradient in the direction `dir=(0|1)`
#   
#   - `description` is a string describing the result returned by `evaluate`. This is helpful because the function may be a cascade of operations. It very much helps to provide short, useful names for your mesh variables to get back reasonable descriptions. 
#   
# Note: at present no error checking is done for consistency between the mesh provided to evaluate and the one used to store the original data. This is very bad on our part !

from quagmire.tools import meshtools
from quagmire import QuagMesh
from quagmire.mesh import MeshVariable
from quagmire import function as fn
import numpy as np

# ### Working mesh
#
# First we create a basic mesh so that we can define mesh variables and obbtain gradients etc.

# +
minX, maxX = -5.0, 5.0
minY, maxY = -5.0, 5.0,
dx, dy = 0.02, 0.02

from stripy.cartesian_meshes import elliptical_base_mesh_points
epointsx, epointsy, ebmask = elliptical_base_mesh_points(10.0, 7.5, 0.25, remove_artifacts=True)
dm = meshtools.create_DMPlex_from_points(epointsx, epointsy, bmask=ebmask, refinement_levels=1)

mesh = QuagMesh(dm, downhill_neighbours=1)
# -

# ### Basic usage
#
# The functions can be demonstrated on the most basic example the `parameter` which is constant everywhere in the mesh. In fact, these operations work without any reference to the mesh since they are the same at all locations and their gradient is zero everywhere. 

# +
A = fn.parameter(10.0)
B = fn.parameter(100.0)

print("Exp({}) = {}".format(A.value, fn.math.exp(A).evaluate(0.0,0.0)))
print("Exp({}) = {}".format(B.value, fn.math.exp(B).evaluate(0.0,0.0)))

## A is a proper lazy variable too so this is required to work

print("Exp({}) = {}".format(A.evaluate(0.0,0.0), fn.math.exp(A).evaluate(0.0,0.0)))

## And this is how to update A

A.value = 100.0
print("Exp({}) = {}".format(A.evaluate(0.0,0.0), fn.math.exp(A).evaluate(0.0,0.0)))

## This works too ... and note the floating point conversion
A(101)
print("Exp({}) = {}".format(A.evaluate(0.0,0.0), fn.math.exp(A).evaluate(0.0,0.0)))

## More complicated examples
print((fn.math.sin(A)**2.0 + fn.math.cos(A)**2.0).evaluate(0.0,0.0))
# -

# ### Descriptions
#
# The lazy function carries a description string that tells you approximately what will happen when the function is evaluated. For example

# +
print(A.description)
print((fn.math.sin(A)+fn.math.cos(B)).description)
print((fn.math.sin(A)**2.0 + fn.math.cos(A)**2.0).description)

## the description is printed by default if you call print on the function 

print((fn.math.sin(A)**2.0 + fn.math.cos(A)**2.0))


# -

# ### Mesh Variables as functions
#
# Mesh variables (`MeshVariables`) are also members of the `LazyEvaluation` class. They can be evaluated exactly as the paramters can, but it is also possible to obtain derivatives. Of course, they also have other properties beyond those of simple functions (see the MeshVariables examples in the XXXXX notebook for details).
#
# Let us first define a mesh variable ... 

height = mesh.add_variable(name="h(X,Y)", lname="h")
height.data = np.ones(mesh.npoints)
print(height)
display(height)

# We might introduce a universal scaling for the height variable. This could be useful if, say, the offset is something that we might want to change programmatically within a timestepping loop. 

# +
h_scale = fn.parameter(2.0)
h_offset = fn.parameter(1.0)

scaled_height = h_scale * height + h_offset

display(scaled_height)

print(height.evaluate(mesh))
print(scaled_height.evaluate(mesh))

h_offset.value = 10.0
print(scaled_height.evaluate(mesh))
# -

# We might wish to define a rainfall parameter that is a function of height that can be passed in to some existing code. The use of functions is perfect for this. 

rainfall_exponent = fn.parameter(2.2)
rainfall = scaled_height**rainfall_exponent
fn.display(rainfall)
print(rainfall.evaluate(mesh))

# The rainfall definition is live to any changes in the height but we can also adjust the rainfall parameters on the fly.
# This allows us to define operators with coefficients that can be supplied as mutable parameters.

# +
height.data = np.sin(mesh.coords[:,0])
print("Height:", height.data)
print("Rainfall Fn evaluation:",rainfall.evaluate(mesh))
print("Rainfall Direct       :",(height.data*2.0+10.0)**2.2)

# change definition of rainfall coefficient but not functional form

rainfall_exponent.value = 1.0
print("Rainfall Fn evaluation:",rainfall.evaluate(mesh))
print("Rainfall Direct       :",(height.data*2.0+10.0)**2.2)
# -

# While functions are most useful because they are not computed once and for all, it is also possible to compute their values and assign to a variable. Just be aware that, at this point, numpy has  a greater richness of operators than `quagmire.function`. We can rewrite the above assignment to the height variable using the `coord` function that extracts values of the x or y ( 0 or 1 ) coordinate direction from the locations given as arguments to `evaluate`. Note that the rainfall function always used the updated height values.

# +
height.data = fn.math.cos(fn.misc.coord(0)).evaluate(mesh)
print("Height:  ", height.data)
print("Height = ", fn.math.cos(fn.misc.coord(0)).description)

rainfall.evaluate(mesh)
# -

# ### Operator overloading for +, - , *, **, /
#
# We define addition / subtraction (negation), multiplication, division, and raising to arbitrary power for mesh variables and parameters and the meaning is carried over from `numpy` - i.e. generally these are element-by-element operations on the underlying data vector and require the data structures to have compatible sizes.
#
# It is common to compute a power law of the magnitude of the local slope. 
#
# $$
#  k = \left| \nabla h \right|^a
# $$

# +
dhdx, dhdy = fn.math.grad(height)
slope = fn.math.sqrt((dhdx**2 + dhdy**2))
a = fn.parameter(1.3)
k = slope**a

display(dhdx)
display(dhdy)
display(k)
print(k)

print(k.evaluate(mesh))
# -


# ### Gradients
#
# Variables associated with a mesh also have the capacity to form spatial derivatives anywhere. This is provided by the `stripy` gradient routines in the case of triangulations. The gradient can be formed from any lazy function by evaluating it at the mesh points and then obtaining values of derivatives anywhere via stripy. In the case of the spatially invariant `parameter` objects, the derivatives are identically zero.
#
# _Note: there is no symbolic differentiation in the functions module._

dhdx = height.fn_gradient(0)
print(dhdx.description)
dh1dy = scaled_height.fn_gradient(1)
print(dh1dy.description)

# Gradients are also accessible through the `grad`, `div` and `curl` operators as follows

gradx, grady = fn.math.grad(height)
div_grad_h = fn.math.div(gradx, grady)
curl_grad_h = fn.math.curl(gradx, grady)
print("Div.Grad(h)  = ", div_grad_h.description)
print("Curl.Grad(h) = ", curl_grad_h.description)

# The following should all evaluate to zero everywhere and so act as a test on the accuracy of the gradient operator 

print("dhdX (error) = ", (gradx+fn.math.sin(fn.misc.coord(0))).evaluate(mesh))
print("dhdY (error) = ",  grady.evaluate(mesh))
print("Curl.Grad(h) (error) = ", curl_grad_h.evaluate(mesh))

# +
import lavavu

xyz     = np.column_stack([mesh.tri.points, height.data])

lv = lavavu.Viewer(border=False, background="#FFFFFF", resolution=[1000,600], near=-10.0)

tris = lv.triangles("triangles",  wireframe=False, colour="#77ff88", opacity=1.0)
tris.vertices(xyz)
tris.indices(mesh.tri.simplices)
tris.values(height.data, label="height")
tris.values(slope.evaluate(mesh), label="slope")
tris.values(curl_grad_h.evaluate(mesh), label="curlgrad")
tris.values(grady.evaluate(mesh), label="dh/dy")

tris.colourmap("elevation")
cb = tris.colourbar()

lv.control.Panel()
lv.control.Range('specular', range=(0,1), step=0.1, value=0.4)
lv.control.Checkbox(property='axis')
lv.control.ObjectList()
tris.control.Checkbox(property="wireframe")
tris.control.List(options=["height", "slope", "curlgrad", "dh/dy"], property="colourby", value="slope", command="redraw", label="Display:")
lv.control.show()
# -

# ### Functions for conditional behaviour
#
# We provide `quagmire.function.misc.where` to produce simple mask functions that can be used to create conditionals. 
# This is how to pick out a flat area in the mesh:
#
# ```python
# flat_area_mask = fn.misc.where(mesh.slope-0.01, fn.parameter(1.0), fn.parameter(0.0)
# ```
#
# The mesh has a mesh.mask variable that is used to identify boundary points. Others could be added (by you) to identify regions such as internal drainages that require special treatment or exclusion from some equations. The levelset function can be applied to a mask to ensure that interpolation does not produce anomalies. It could also be used to clip out a value in a field between certain ranges (e.g. to capture regions in a specific height interval or with a specific catchment identifier). 
#

masked_height = fn.misc.where(fn.misc.coord(1), height, 0.0)
masked_height.display()
masked_height.derivative(1)
print(masked_height)

flat_area_mask  = fn.misc.where(0.1-slope, 1.0, 0.0 )
steep_area_mask = fn.misc.where(slope-0.9, 1.0, 0.0 )
flat_area_mask.display()
steep_area_mask.display()

# +
import lavavu

xyz     = np.column_stack([mesh.tri.points, masked_height.evaluate(mesh)])

lv = lavavu.Viewer(border=False, background="#FFFFFF", resolution=[1000,600], near=-10.0)

tris = lv.triangles("triangles",  wireframe=False, colour="#77ff88", opacity=1.0)
tris.vertices(xyz)
tris.indices(mesh.tri.simplices)
tris.values(height.data, label="height")
tris.values(steep_area_mask.evaluate(mesh), label="steep")
tris.values(flat_area_mask.evaluate(mesh), label="flat")


tris.colourmap("elevation")
cb = tris.colourbar()

lv.control.Panel()
lv.control.Range('specular', range=(0,1), step=0.1, value=0.4)
lv.control.Checkbox(property='axis')
lv.control.ObjectList()
tris.control.Checkbox(property="wireframe")
tris.control.List(options=["height", "steep", "flat"], property="colourby", value="flat", command="redraw", label="Display:")
lv.control.show()
# -
# Note how the derivative of the 'level set' functions works. We assume that the derivative of the 
# masking function is zero everywhere so that the mask simply applies to the derivative of the masked 
# function:


masked_height.derivative(1)
