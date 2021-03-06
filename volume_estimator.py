
import numpy as np
import pandas as pd
import cv2
import json
from scipy.spatial.distance import pdist
from scipy.stats import skew
from keras.models import Model, model_from_json
import keras.backend as K
import matplotlib.pyplot as plt
from VolumeCalculationUtils import get_cloud , sor_filter ,pca_plane_estimation ,align_plane_with_axis , pc_to_volume
from DepthAPI.depth import dpt
from SegmentationAPI.segmentation import utils , SeMask_FPN 
from SegmentationAPI.api import segmentation_inference
from coin_detector import CoinDetector
import tensorflow as tf
import open3d as  o3d


class VolumeEstimator():
    def __init__(self,fov =70,model_input_shape = (384,384),relax_param =0.01):

        self.fov = fov
        self.model_input_shape = model_input_shape
        self.relax_param = relax_param
        self.coinDetector = CoinDetector()
        self.coin_diameter_prior = 2.6  # 10 Baht diameter


    def get_depth_from_image(self,input_image_rgb):
        # Predict depth by receive input image in rgb format and return result of depth map in numpy array in meter unit.
        
        inverse_depth = dpt.predict(input_image_rgb)

        # disparity_map = (self.min_disp + (self.max_disp - self.min_disp) 
        #                  * inverse_depth)

        depth = 1 / np.asarray(inverse_depth)
        return depth 

    def get_point_cloud(self, input_image_bgr, fov=70):
        # Load input image and resize to model input size
        img = input_image_bgr
        input_image_shape = img.shape
        print(input_image_shape)
        # img = cv2.resize(img, (self.model_input_shape[1],
        #                        self.model_input_shape[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # now img is in rbg format

        # Create intrinsics matrix
        intrinsics_mat = self.__create_intrinsics_matrix(input_image_shape, fov)
        intrinsics_inv = np.linalg.inv(intrinsics_mat)

        # Predict depth
        

        # disparity_map = (self.min_disp + (self.max_disp - self.min_disp) 
        #                  * inverse_depth)

        depth = self.get_depth_from_image(img)


        # depth is now numpy 
        # Convert depth map to point cloud
        depth_tensor = K.variable(np.expand_dims(depth, 0))
        intrinsics_inv_tensor = K.variable(np.expand_dims(intrinsics_inv, 0))
        point_cloud = K.eval(get_cloud(depth_tensor, intrinsics_inv_tensor))
        print("Done createing point_cloud")
        point_cloud_flat = np.reshape( point_cloud, (point_cloud.shape[1] * point_cloud.shape[2], 3))

        print(point_cloud)
        print(point_cloud.shape)
        print(point_cloud_flat.shape)



    def get_segmentation_mask(self,input_image_rgb):
        # Predict boolean mask of all ingredient found in image 
        input_image_bgr = cv2.cvtColor(input_image_rgb,cv2.COLOR_RGB2BGR)
        filter_thresold = 0.001
        mask = np.asarray(SeMask_FPN.predict(input_image_bgr))
        food_types = np.unique(mask)

        masks_array = []

        masks_type = []
        

        for ingredient in food_types:
            # filter background
            mask_zero_one = (mask==ingredient).astype(int)
            if ingredient != 0 and np.count_nonzero(mask_zero_one) / (mask_zero_one.shape[0]*mask_zero_one.shape[1])>filter_thresold:
                masks_array.append(mask_zero_one)
                masks_type.append(utils.id2label[ingredient])
        print('[*] Found {} food object(s) ''in image.'.format(len(masks_type)))
        return masks_array , masks_type


    
    def test_scaling(self,input_image_bgr):
        ellipse_params_scaled= self.get_ellipse_params(input_image_bgr)
        if (len(ellipse_params_scaled)==0):
            print("don't detect coin")
            return 
        img0 = cv2.resize(input_image_bgr, (self.model_input_shape[1],self.model_input_shape[0]))

        coin_point_1 = [ int(ellipse_params_scaled[2] 
                            * np.cos(ellipse_params_scaled[4]) 
                            + ellipse_params_scaled[0]),int(ellipse_params_scaled[2] 
                            * np.sin(ellipse_params_scaled[4]) 
                            + ellipse_params_scaled[1])]
        coin_point_2 = [int(-ellipse_params_scaled[2] 
                            * np.cos(ellipse_params_scaled[4]) 
                            + ellipse_params_scaled[0]),int(-ellipse_params_scaled[2] 
                            * np.sin(ellipse_params_scaled[4]) 
                            + ellipse_params_scaled[1])]
        print(ellipse_params_scaled)
        img0 = cv2.ellipse(img0, (int(ellipse_params_scaled[0]),int(ellipse_params_scaled[1])), (int(ellipse_params_scaled[2]),int(ellipse_params_scaled[3])),int(ellipse_params_scaled[4]),0 , 360, (62, 3, 255), 2)

        img0 = cv2.line(img0,tuple(coin_point_1),tuple(coin_point_2),(255,0,255),1)
        img0 = cv2.circle(img0, tuple(coin_point_1), radius=1, color=(0, 255, 0), thickness=-1)
        img0 = cv2.circle(img0, tuple(coin_point_2), radius=1, color=(0, 255, 0), thickness=-1)
        cv2.imshow('window', img0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_ellipse_params(self,img_bgr):
        # Find ellipse parameterss (cx, cy, a, b, theta) that 
        # describe the coin 
        print("image shape ",img_bgr.shape)
        ellipse_scale = [self.model_input_shape[0]/img_bgr.shape[0],self.model_input_shape[1]/img_bgr.shape[1]]
        ellipse_params_list = self.coinDetector.predict(img_bgr)

        # y heigth shape[0]
        # x weigth shape[1]
        if(len(ellipse_params_list) == 0):
            return ellipse_params_list

        ellipse_params = ellipse_params_list[0]

        ellipse_params[0] = ellipse_params[0] *ellipse_scale[1]  #cx
        ellipse_params[1] = ellipse_params[1] *ellipse_scale[0]  #cy
        ellipse_params[2] = ellipse_params[2] *ellipse_scale[1]  #a
        ellipse_params[3] = ellipse_params[3] *ellipse_scale[0]  #b

        return ellipse_params



    def scale_with_coin(self,img_bgr,point_cloud):
       # Scale depth map with coin detection (cx, cy, a, b, theta) 
        scaling = 100

        
        ellipse_params_scaled= self.get_ellipse_params(img_bgr)
        if len(ellipse_params_scaled)  == 0:
            print("[ ] coin detector can not found coin use original depth map")
            return scaling


        coin_point_1 = [int(ellipse_params_scaled[2] 
                            * np.sin(ellipse_params_scaled[4]) 
                            + ellipse_params_scaled[1]), 
                            int(ellipse_params_scaled[2] 
                            * np.cos(ellipse_params_scaled[4]) 
                            + ellipse_params_scaled[0])]
        coin_point_2 = [int(-ellipse_params_scaled[2] 
                            * np.sin(ellipse_params_scaled[4]) 
                            + ellipse_params_scaled[1]),
                            int(-ellipse_params_scaled[2] 
                            * np.cos(ellipse_params_scaled[4]) 
                            + ellipse_params_scaled[0])]
        coin_point_1_3d = point_cloud[0, coin_point_1[0], 
                                        coin_point_1[1], :]
        coin_point_2_3d = point_cloud[0, coin_point_2[0], 
                                        coin_point_2[1], :]
        coin_diameter = np.linalg.norm(coin_point_1_3d 
                                        - coin_point_2_3d)
        scaling = self.coin_diameter_prior / coin_diameter


        print("[scaling coin ] : ",scaling)


        return scaling
  


    def estimate_volume(self, input_image_bgr, fov=70,coin_scale = False):
        """Volume estimation procedure.

        Inputs:
            input_image: Path to input image or image array.
            fov: Camera Field of View.
            plate_diameter_prior: Expected plate diameter.
            plot_results: Result plotting flag.
            plots_directory: Directory to save plots at or None.
        Returns:
            estimated_volume: Estimated volume.
            an list of volume of every ingredient
        """
        # Load input image and resize to model input size
        img = input_image_bgr
        input_image_shape = img.shape
        print("input_image_shape" ,input_image_shape)
        img = cv2.resize(img, (self.model_input_shape[1],
                               self.model_input_shape[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        intrinsics_mat = self.__create_intrinsics_matrix(input_image_shape, fov)
        intrinsics_inv = np.linalg.inv(intrinsics_mat)
        depth = self.get_depth_from_image(img)
        # depth is now numpy 
        # Convert depth map to point cloud
        depth_tensor = K.variable(np.expand_dims(depth, 0))
        intrinsics_inv_tensor = K.variable(np.expand_dims(intrinsics_inv, 0))
        point_cloud = K.eval(get_cloud(depth_tensor, intrinsics_inv_tensor))
        print("Done createing point_cloud")
        point_cloud_flat = np.reshape( point_cloud, (point_cloud.shape[1] * point_cloud.shape[2], 3))
        print("point_cloud -->",point_cloud[0,1,1])
        print(point_cloud.shape)
        print(point_cloud_flat.shape)
        # Scale depth map with coin detection
        print("Use sacle ?",coin_scale)
        if coin_scale:
            print("Use scale")
            scaling = self.scale_with_coin(input_image_bgr,point_cloud)
            depth = scaling * depth
            point_cloud = scaling * point_cloud
            point_cloud_flat = scaling * point_cloud_flat
        else:
            scaling = 100
            depth = scaling * depth
            point_cloud = scaling * point_cloud
            point_cloud_flat = scaling * point_cloud_flat

        # Predict segmentation masks
        masks_array , masks_type = self.get_segmentation_mask(img)


        print("Done predict segmentation masks")

        # Iterate over all predicted masks and estimate volumes
        estimated_volumes = {}
        total_volume = 0
        for k in range(len(masks_array)):
            print(masks_type[k])
            # Apply mask to create object image and depth map
            object_mask = masks_array[k]
            object_img = (np.tile(np.expand_dims(object_mask, axis=-1),(1,1,3)) * img)
            object_depth = object_mask * depth
            print("Done Apply mask to create object image and depth map")
            # Get object/non-object points by filtering zero/non-zero 
            # depth pixels
    
            object_mask = (np.reshape(object_depth, (object_depth.shape[0] * object_depth.shape[1])) > 0)
            
            object_points = point_cloud_flat[object_mask, :]
            
            # non_object_points = point_cloud_flat[np.logical_not(object_mask), :]
            



            print("Done create object point")
            # print(object_points)
            print(object_points.shape)


            # Filter outlier points
            object_points_filtered = object_points


            print("Done Filter outlier points")
            # Estimate base plane parameters
            plane_params = pca_plane_estimation(object_points_filtered)
            print("Done Estimate base plane parameters")
            # Transform object to match z-axis with plane normal
            translation, rotation_matrix = align_plane_with_axis(plane_params, np.array([0, 0, 1]))
            object_points_transformed = np.dot(object_points_filtered + translation, rotation_matrix.T)
            print("Done Transform object to match z-axis with plane normal")

            # Adjust object on base plane
            height_sorted_indices = np.argsort(object_points_transformed[:,2])
            adjustment_index = height_sorted_indices[int(object_points_transformed.shape[0] * self.relax_param)]
            object_points_transformed[:,2] += np.abs(object_points_transformed[adjustment_index, 2])
            
            print("Done Adjust object on base plane")


            volume_points = object_points_transformed[
            object_points_transformed[:,2] > 0]
            estimated_volume, _ = pc_to_volume(volume_points)
            print("Done calculate volume from point clound")

            estimated_volumes[masks_type[k]] = estimated_volume
            total_volume += estimated_volume
             

        estimated_volumes["total"] = total_volume
        return estimated_volumes

    def visualization_volume(self, input_image_bgr, fov=70,coin_scale=True,mark_coin_point =True,alpha_list = [0.01]):
        img = input_image_bgr
        input_image_shape = img.shape
        print("input_image_shape" ,input_image_shape)
        img = cv2.resize(img, (self.model_input_shape[1],
                            self.model_input_shape[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        intrinsics_mat = self.__create_intrinsics_matrix(input_image_shape, fov)
        intrinsics_inv = np.linalg.inv(intrinsics_mat)
        depth = self.get_depth_from_image(img)
        depth_tensor = K.variable(np.expand_dims(depth, 0))
        intrinsics_inv_tensor = K.variable(np.expand_dims(intrinsics_inv, 0))
        point_cloud = K.eval(get_cloud(depth_tensor, intrinsics_inv_tensor))
        print("Done createing point_cloud")

        point_cloud_flat = np.reshape( point_cloud, (point_cloud.shape[1] * point_cloud.shape[2], 3))
        print("point_cloud -->",point_cloud.shape)
        print("flat_point_cloud -->",point_cloud_flat.shape)

        print("Use sacle ?",coin_scale)
        if coin_scale:
            print("Use scale")
            scaling = self.scale_with_coin(input_image_bgr,point_cloud)
            depth = scaling * depth
            point_cloud = scaling * point_cloud
            point_cloud_flat = scaling * point_cloud_flat

            if mark_coin_point:
                print("marking_point_coin")
                        #mark coin point 

                ellipse_params_scaled= self.get_ellipse_params(input_image_bgr)
                coin_point_1 = [int(ellipse_params_scaled[2] 
                                    * np.sin(ellipse_params_scaled[4]) 
                                    + ellipse_params_scaled[1]), 
                                    int(ellipse_params_scaled[2] 
                                    * np.cos(ellipse_params_scaled[4]) 
                                    + ellipse_params_scaled[0])]
                coin_point_2 = [int(-ellipse_params_scaled[2] 
                                    * np.sin(ellipse_params_scaled[4]) 
                                    + ellipse_params_scaled[1]),
                                    int(-ellipse_params_scaled[2] 
                                    * np.cos(ellipse_params_scaled[4]) 
                                    + ellipse_params_scaled[0])]
                img[coin_point_1[0],coin_point_1[1] ]= [235, 52, 225]
                img[coin_point_2[0],coin_point_2[1] ]= [235, 52, 225]
        else:
            #convert from meter to centemer
            scaling = 100
            depth = scaling * depth
            point_cloud = scaling * point_cloud
            point_cloud_flat = scaling * point_cloud_flat



        image_color_flat = np.reshape( img ,(img.shape[0] * img.shape[1], 3))/255
        print("image flat --->",image_color_flat.shape)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud_flat)
        pcd.colors = o3d.utility.Vector3dVector(image_color_flat)
        o3d.visualization.draw_geometries([pcd])

        for alpha in alpha_list:
            print(f"alpha={alpha:.3f}")
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
            mesh.compute_vertex_normals()
            o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    def visualization_point_clound(self, input_image_bgr, fov=70,coin_scale=True,mark_coin_point =True):
        img = input_image_bgr
        input_image_shape = img.shape
        print("input_image_shape" ,input_image_shape)
        img = cv2.resize(img, (self.model_input_shape[1],
                            self.model_input_shape[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        intrinsics_mat = self.__create_intrinsics_matrix(input_image_shape, fov)
        intrinsics_inv = np.linalg.inv(intrinsics_mat)
        depth = self.get_depth_from_image(img)
        depth_tensor = K.variable(np.expand_dims(depth, 0))
        intrinsics_inv_tensor = K.variable(np.expand_dims(intrinsics_inv, 0))
        point_cloud = K.eval(get_cloud(depth_tensor, intrinsics_inv_tensor))
        print("Done createing point_cloud")

        point_cloud_flat = np.reshape( point_cloud, (point_cloud.shape[1] * point_cloud.shape[2], 3))
        print("point_cloud -->",point_cloud.shape)
        print("flat_point_cloud -->",point_cloud_flat.shape)

        print("Use sacle ?",coin_scale)
        if coin_scale:
            print("Use scale")
            scaling = self.scale_with_coin(input_image_bgr,point_cloud)
            depth = scaling * depth
            point_cloud = scaling * point_cloud
            point_cloud_flat = scaling * point_cloud_flat

            if mark_coin_point:
                print("marking_point_coin")
                        #mark coin point 

                ellipse_params_scaled= self.get_ellipse_params(input_image_bgr)
                coin_point_1 = [int(ellipse_params_scaled[2] 
                                    * np.sin(ellipse_params_scaled[4]) 
                                    + ellipse_params_scaled[1]), 
                                    int(ellipse_params_scaled[2] 
                                    * np.cos(ellipse_params_scaled[4]) 
                                    + ellipse_params_scaled[0])]
                coin_point_2 = [int(-ellipse_params_scaled[2] 
                                    * np.sin(ellipse_params_scaled[4]) 
                                    + ellipse_params_scaled[1]),
                                    int(-ellipse_params_scaled[2] 
                                    * np.cos(ellipse_params_scaled[4]) 
                                    + ellipse_params_scaled[0])]
                img[coin_point_1[0],coin_point_1[1] ]= [235, 52, 225]
                img[coin_point_2[0],coin_point_2[1] ]= [235, 52, 225]



        image_color_flat = np.reshape( img ,(img.shape[0] * img.shape[1], 3))/255
        print("image flat --->",image_color_flat.shape)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud_flat)
        pcd.colors = o3d.utility.Vector3dVector(image_color_flat)
        o3d.visualization.draw_geometries([pcd])

    def __create_intrinsics_matrix(self, input_image_shape, fov):
        """Create intrinsics matrix from given camera fov.

        Inputs:
            input_image_shape: Original input image shape.
            fov: Camera Field of View (in deg).
        Returns:
            intrinsics_mat: Intrinsics matrix [3x3].
        """
        F = input_image_shape[1] / (2 * np.tan((fov / 2) * np.pi / 180))
        print('[*] Creating intrinsics matrix from given FOV:', fov)

        # Create intrinsics matrix
        x_scaling = int(self.model_input_shape[1]) / input_image_shape[1] 
        y_scaling = int(self.model_input_shape[0]) / input_image_shape[0] 
        intrinsics_mat = np.array(
            [[F * x_scaling, 0, (input_image_shape[1] / 2) * x_scaling], 
             [0, F * y_scaling, (input_image_shape[0] / 2) * y_scaling],
             [0, 0, 1]])
        return intrinsics_mat




if __name__ == '__main__':
    volumeEstimator = VolumeEstimator()
    img = cv2.imread("imgs/test1.jpg")
    print("image size ",img.shape)
    # volumeEstimator.
    # for fov in [20,30,50,70]:
    #     print("fov ---> ",fov)
    #     volumeEstimator.visualization_point_clound(img,fov,True,True)
    # volumeEstimator.visualization_point_clound(img,70,True,False)
    volumeEstimator.visualization_volume(img,70,True,True,[1.0])