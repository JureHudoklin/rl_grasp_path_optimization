import numpy as np
import trimesh


class VisualizationObject():
    def __init__(self):
        self.meshes = []

    def display(self, my_scene = None, clear_meshes = True, add_lights = True):
        """
        Plots all the meshes that were cached.
        --------------
        clear_meshes : Bool
            If true resets all the cashed meshes after plotting
        
        Return
        --------------
        None : None
        """
        # Create the scene
        if my_scene == None:
            my_scene = trimesh.scene.scene.Scene(self.meshes)
        else:
            for mesh in self.meshes:
                my_scene.add_geometry(mesh)

        # Add a specific light
        if add_lights == True:
            light_tr_MX = trimesh.transformations.translation_matrix(np.array([0, 100, 10]))
            lights = trimesh.scene.lighting.DirectionalLight(
                name="light_1", color=[100, 100, 100, 255], intensity=5.0, radius=1)
            my_scene.graph[lights.name] = light_tr_MX
            my_scene.lights = [lights]

        # Plot scene
        my_scene.show()

        if clear_meshes == True:
            self.meshes = []

            

    def plot_point(self, point, color = [255, 0, 0, 255], radius = 1):
        """
        Plots a point 
        --------------
        point : np.array(3,)
            3D coordinates of the point
        color : array(4,)
            Color of the point: [R,G,B, Intensity]
        radius : float
            Radius of the displayed point
        
        Return
        --------------
        None : None
        """
        mesh_point = trimesh.primitives.Sphere(radius=radius)
        mesh_point.apply_translation(point)
        mesh_point.visual.face_colors = color
        self.meshes.append(mesh_point)

    def plot_point_multiple(self, points, color=[255, 0, 0, 255], radius=0.5):
        """
        Plots points
        --------------
        points : np.array(n,3)
            3D coordinates of the point
        color : array(4,)
            Color of the point: [R,G,B, Intensity]
        radius : float
            Radius of the displayed point
        
        Return:
        --------------
        None : None
        """
        for point in points:
            mesh_point = trimesh.primitives.Sphere(radius=radius)
            mesh_point.apply_translation(point)
            mesh_point.visual.face_colors = color
            self.meshes.append(mesh_point)

    def plot_point_cloud(self, pc, color=[255, 0, 0, 255], radius=0.5):
        """
        Plots a point cloud
        --------------
        pc : np.array(n,3)
            Point cloud
        color : array(4,)
            Color of the point: [R,G,B, Intensity]
        radius : float
            Radius of the displayed point
        
        Return:
        --------------
        None : None
        """
        pc_mesh = trimesh.points.PointCloud(pc, colors=color, radius=radius)
        self.meshes.append(pc_mesh)

    def plot_mesh(self, mesh, color=[150, 150, 150, 255], **kwargs):
        """
        Plots the mesh given a trimesh mesh
        --------------
        Arguments:
            mesh : trimesh.Trimesh
                Mesh to be plotted
            color : array(4,)
                Color of the point: [R,G,B, Intensity]
        Return:
        --------------
            None : None
        """
        # Unmerge so viewer doesn't smooth
        #mesh.unmerge_vertices()
        # Assign color
        #mesh.visual.face_colors = color
        # Append to meshes
        self.meshes.append(mesh)

    def plot_object(self, obj_model, color=[150, 150, 150, 255], **kwargs):
        """
        Plots the object given object model
        --------------
        obj_model : ObjectModel
            Model contactining trimesh
        color : array(4,)
            Color of the point: [R,G,B, Intensity]
        
        Return:
        --------------
        None : None
        """
        if "file_loc" in kwargs:
            mesh = trimesh.load(kwargs["file_loc"])
            if "units" in kwargs:   # Convert mesh to correct units
                mesh.units = kwargs["units"][0]
                mesh.convert_units(kwargs["units"][1])

        else:
            mesh = obj_model.mesh

        # Unmerge so viewer doesn't smooth
        mesh.unmerge_vertices()
        # Assign color
        mesh.visual.face_colors = color
        # Append to meshes
        self.meshes.append(mesh)


    def plot_vector(self, p1, p2, color=[0, 0, 0, 255], radius_cyl=0.6, arrow=True):
        """
        Plats a vector from point 1 to point 2
        --------------
        p1 : np.array(3,)
            3D coordinates of point 1
        p2 : np.array(3,)
            3D coordinates of point 2
        color : array(4,)
            Color of the point: [R,G,B, Intensity]
        radius_cyl : float
            The size of the arrow
        arrow : Bool
            If true arrow is displayed at the end of the vector, otherwise just a line
        
        Return:
        --------------
        None : None
        """
        vector = p2-p1
        length = np.linalg.norm(p2-p1)
        if length < 0.0001:
            return
        if arrow:
            height_cylinder = length * 8/10
            height_cone = length * 2/10
        else:
            height_cylinder = length
            height_cone = length * 2/10
        rot_MX = trimesh.geometry.align_vectors(np.array([0, 0, 1]), vector)
        tr_MX = trimesh.transformations.translation_matrix(p1)

        # Create Cylinder
        cylinder = trimesh.primitives.Cylinder(
            radius=radius_cyl, height=height_cylinder, sections = 3)
        cylinder.visual.face_colors = color
        cylinder.apply_translation(np.array([0, 0, height_cylinder/2]))
        cylinder.apply_transform(rot_MX)
        cylinder.apply_transform(tr_MX)

        # Create Cone
        cone = trimesh.creation.cone(radius=radius_cyl*1.5, height=height_cone)
        cone.visual.face_colors = color
        tr_cone = trimesh.transformations.translation_matrix(
            np.array([0, 0, height_cylinder]))
        cone.apply_transform(tr_cone)
        cone.apply_transform(rot_MX)
        cone.apply_transform(tr_MX)

        self.meshes.append(cylinder)
        if arrow:
            self.meshes.append(cone)

    def plot_grasp(self, grasp_tf, score = 1, units = "millimeters"):
        """
        Given grasp transform matrices and their score, plots the grasp
        --------------
        Arguments:
            grasp_tf : np.array(4,4)
                Transformation matrix
            score : np.array(n)
                Scores of individual grasps
        Return:
        --------------
            None : None
        """

        if units == "millimeters":
            length_coef = 10
        else:
            length_coef = 0.01
        if len(grasp_tf) == 1:
            grasp_point = np.array([0, 0, 0])
            grasp_dir = np.array([0, 0,length_coef*3*score])
            points_transformed = trimesh.transform_points([grasp_point, grasp_dir], grasp_tf)
            grasp_point = np.array(points_transformed[0])
            grasp_dir = np.array(points_transformed[1])
            self.plot_vector(grasp_point, grasp_dir, radius_cyl=length_coef/10, arrow=False)
        else:
            for i in range(len(grasp_tf)):
                if round(score[i], 4) == 0:
                    continue
                grasp_point = np.array([0, 0, 0])
                grasp_dir = np.array([0, 0, length_coef*3*score[i]])
                points_transformed = trimesh.transform_points([grasp_point, grasp_dir], grasp_tf[i])
                grasp_point = np.array(points_transformed[0])
                grasp_dir = np.array(points_transformed[1])
                self.plot_vector(grasp_point, grasp_dir,
                                 radius_cyl=length_coef/10, arrow=False)

    def plot_coordinate_system(self, tf=None, scale=1):
        """
        Plot a coordinate system given a transformation matrix and a scale
        --------------
        Arguments:
            tf : np.array(4,4)
                Transformation matrix
            scale : float
                Scale of the coordinate system
        Return:
        --------------
            None : None
        """
        if type(tf) == None:
            tf = np.eye(4)
        cs = trimesh.creation.axis(origin_size=scale, transform=tf)
        self.meshes.append(cs)


    def plot_points_score(self, results):
        print(results.keys())

        for i in range(len(results["p_0"])):
            mesh_point = trimesh.primitives.Sphere(radius=1)
            mesh_point.apply_translation(results["p_0"][i])
            if results["score_total"][i] > 0:
                score = 1
            else:
                score = 0
            mesh_point.visual.face_colors = [
                int((1-results["score_total"][i])*255), int(score*255), 0, 255]
            self.meshes.append(mesh_point)




