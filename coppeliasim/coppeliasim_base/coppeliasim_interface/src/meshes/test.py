from __future__ import print_function
import trimesh
import trimesh_visualization as tv

my_scene = tv.VisualizationObject()
my_scene.plot_coordinate_system(scale=0.01)
my_scene.plot_mesh(trimesh.load("object_0.obj"))

my_scene.display()
