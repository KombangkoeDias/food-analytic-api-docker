
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


class VolumeEstimator():
    def __init__(self,fov =70,model_input_shape = (384,384),relax_param =0.01):

        self.fov = fov
        self.model_input_shape = model_input_shape
        self.relax_param = relax_param


    def get_depth_from_image(self,input_image_rgb):
        # Predict depth by receive input image in rgb format and return result of depth map in numpy array in meter unit.
        
        inverse_depth = dpt.predict(input_image_rgb)

        # disparity_map = (self.min_disp + (self.max_disp - self.min_disp) 
        #                  * inverse_depth)

        depth = 1 / np.asarray(inverse_depth)

        

        return depth 

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


    def estimate_volume(self, input_image_bgr, fov=70, ):
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
        img = cv2.resize(img, (self.model_input_shape[1],
                               self.model_input_shape[0]))
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

        # print(point_cloud)
        print(point_cloud.shape)
        print(point_cloud_flat.shape)




        # Scale depth map with coin detection

        # Find ellipse parameterss (cx, cy, a, b, theta) that 
        # describe the plate contour
        # ellipse_scale = 2
        # ellipse_detector = EllipseDetector(
        #     (ellipse_scale * self.model_input_shape[0],
        #      ellipse_scale * self.model_input_shape[1]))
        # ellipse_params = ellipse_detector.detect(input_image)
        # ellipse_params_scaled = tuple(
        #     [x / ellipse_scale for x in ellipse_params[:-1]]
        #     + [ellipse_params[-1]])
        # if (any(x != 0 for x in ellipse_params_scaled) and
        #         plate_diameter_prior != 0):
        #     print('[*] Ellipse parameters:', ellipse_params_scaled)
        #     # Find the scaling factor to match prior 
        #     # and measured plate diameters
        #     plate_point_1 = [int(ellipse_params_scaled[2] 
        #                      * np.sin(ellipse_params_scaled[4]) 
        #                      + ellipse_params_scaled[1]), 
        #                      int(ellipse_params_scaled[2] 
        #                      * np.cos(ellipse_params_scaled[4]) 
        #                      + ellipse_params_scaled[0])]
        #     plate_point_2 = [int(-ellipse_params_scaled[2] 
        #                      * np.sin(ellipse_params_scaled[4]) 
        #                      + ellipse_params_scaled[1]),
        #                      int(-ellipse_params_scaled[2] 
        #                      * np.cos(ellipse_params_scaled[4]) 
        #                      + ellipse_params_scaled[0])]
        #     plate_point_1_3d = point_cloud[0, plate_point_1[0], 
        #                                    plate_point_1[1], :]
        #     plate_point_2_3d = point_cloud[0, plate_point_2[0], 
        #                                    plate_point_2[1], :]
        #     plate_diameter = np.linalg.norm(plate_point_1_3d 
        #                                     - plate_point_2_3d)
        #     scaling = plate_diameter_prior / plate_diameter
        # else:
        #     # Use the median ground truth depth scaling when not using
        #     # the plate contour
        #     print('[*] No ellipse found. Scaling with expected median depth.')
        #     predicted_median_depth = np.median(1 / disparity_map)
        #     scaling = self.gt_depth_scale / predicted_median_depth
        # depth = scaling * depth
        # point_cloud = scaling * point_cloud
        # point_cloud_flat = scaling * point_cloud_flat

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
            # object_points_filtered, sor_mask = sor_filter(object_points, 2, 0.7)
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
             
            # if (plot_results or plots_directory is not None):
            #     # Create object points from estimated plane
            #     plane_z = np.apply_along_axis( lambda x: ((plane_params[0] + plane_params[1] * x[0]+ plane_params[2] * x[1]) * (-1) / plane_params[3]), axis=1, arr=point_cloud_flat[:,:2])
            #     plane_points = np.concatenate( (point_cloud_flat[:,:2], np.expand_dims(plane_z, axis=-1)), axis=-1)
            #     plane_points_transformed = np.dot(plane_points + translation,  rotation_matrix.T)
            #     print('[*] Estimated plane parameters (w0,w1,w2,w3):',
            #           plane_params)

            #     # Get the color values for the different sets of points
            #     colors_flat = (np.reshape(img, (self.model_input_shape[0]  * self.model_input_shape[1], 3))* 255)
            #     object_colors = colors_flat[object_mask, :]
            #     non_object_colors= colors_flat[np.logical_not(object_mask), :]
            #     object_colors_filtered = object_colors[sor_mask, :]

            #     # Create dataFrames for the different sets of points
            #     non_object_points_df = pd.DataFrame( np.concatenate((non_object_points, non_object_colors), axis=-1),columns=['x','y','z','red','green','blue'])
            #     object_points_df = pd.DataFrame(np.concatenate((object_points, object_colors), axis=-1), columns=['x','y','z','red','green','blue'])
            #     plane_points_df = pd.DataFrame( plane_points, columns=['x','y','z'])
            #     object_points_transformed_df = pd.DataFrame( np.concatenate((object_points_transformed,object_colors_filtered), axis=-1),
            #         columns=['x','y','z','red','green','blue'])
            #     plane_points_transformed_df = pd.DataFrame(plane_points_transformed, columns=['x','y','z'])

            #     # Outline the detected plate ellipse and major axis vertices 
            #     # plate_contour = np.copy(img)
            #     # if (any(x != 0 for x in ellipse_params_scaled) and
            #     #     plate_diameter_prior != 0):
            #     #     ellipse_color = (68 / 255, 1 / 255, 84 / 255)
            #     #     vertex_color = (253 / 255, 231 / 255, 37 / 255)
            #     #     cv2.ellipse(plate_contour,
            #     #                 (int(ellipse_params_scaled[0]),
            #     #                  int(ellipse_params_scaled[1])), 
            #     #                 (int(ellipse_params_scaled[2]),
            #     #                  int(ellipse_params_scaled[3])),
            #     #                 ellipse_params_scaled[4] * 180 / np.pi, 
            #     #                 0, 360, ellipse_color, 2)
            #     #     cv2.circle(plate_contour,
            #     #                (int(plate_point_1[1]), int(plate_point_1[0])),
            #     #                2, vertex_color, -1)
            #     #     cv2.circle(plate_contour,
            #     #                (int(plate_point_2[1]), int(plate_point_2[0])),
            #     #                2, vertex_color, -1)

            #     # Estimate volume for points above the plane
            #     volume_points = object_points_transformed[ object_points_transformed[:,2] > 0]
            #     estimated_volume, simplices = pc_to_volume(volume_points)
            #     print('[*] Estimated volume:', estimated_volume * 1000, 'L')

            #     # Create figure of input image and predicted 
            #     # plate contour, segmentation mask and depth map
            #     pretty_plotting([img, plate_contour, depth, object_img], 
            #                     (2,2),
            #                     ['Input Image', 'Plate Contour', 'Depth', 
            #                      'Object Mask'],
            #                     'Estimated Volume: {:.3f} L'.format(
            #                     estimated_volume * 1000.0))

            #     # Plot and save figure
            #     if plot_results:
            #         plt.show()
            #     if plots_directory is not None:
            #         if not os.path.exists(plots_directory):
            #             os.makedirs(plots_directory)
            #         (img_name, ext) = os.path.splitext(
            #             os.path.basename(input_image_bgr))
            #         filename = '{}_{}{}'.format(img_name, plt.gcf().number,
            #                                     ext)
            #         plt.savefig(os.path.join(plots_directory, filename))

            #     estimated_volumes.append(
            #         (estimated_volume, object_points_df, non_object_points_df,
            #          plane_points_df, object_points_transformed_df,
            #          plane_points_transformed_df, simplices))
            # else:
                # Estimate volume for points above the plane
                # volume_points = object_points_transformed[
                #     object_points_transformed[:,2] > 0]
                # estimated_volume, _ = pc_to_volume(volume_points)
                # estimated_volumes.append(estimated_volume)

        estimated_volumes["total"] = total_volume
        return estimated_volumes

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
    estimator = VolumeEstimator()

    # Iterate over input images to estimate volumes
    results = {'image_path': [], 'volumes': []}
    for input_image in estimator.args.input_images:
        print('[*] Input:', input_image)
        volumes = estimator.estimate_volume(
            input_image, estimator.args.fov, 
            estimator.args.plate_diameter_prior, estimator.args.plot_results,
            estimator.args.plots_directory)

        # Store results per input image
        results['image_path'].append(input_image)
        if (estimator.args.plot_results 
            or estimator.args.plots_directory is not None):
            results['volumes'].append([x[0] * 1000 for x in volumes])
            plt.close('all')
        else:
            results['volumes'].append(volumes * 1000)

        # Print weight if density database is given
        if estimator.args.density_db is not None:
            db_entry = estimator.density_db.query(
                estimator.args.food_type)
            density = db_entry[1]
            print('[*] Density database match:', db_entry)
            # All foods found in the input image are considered to be
            # of the same type
            for v in results['volumes'][-1]:
                print('[*] Food weight:', 1000 * v * density, 'g')

    if estimator.args.results_file is not None:
        # Save results in CSV format
        volumes_df = pd.DataFrame(data=results)
        volumes_df.to_csv(estimator.args.results_file, index=False)


