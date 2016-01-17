import json
import os

import sys
sys.path.append('/usr/lib/python2.7/dist-packages') 
import cv2

import numpy as np

from tqdm import *

import dataset_utils


class Rovina(object):
    def __init__(self, config_filename):
        self.config_filename = config_filename
        with open(config_filename) as config_file:
            self.config = json.load(config_file)

        if self.config['use_relative_paths']:
            self.root_folder = os.path.dirname(config_filename)
        else:
            self.root_folder = ''

        #Used if we want to use the "flipped" version of the camera 0.
        self.folder_postfix = self.config['flipped_post_fix']
        

        image_f = self.config['image_folder']
        if image_f is not None:
            image_f += self.folder_postfix
            self.image_folder = os.path.join(self.root_folder, image_f)
            self.image_extension = self.config['image_extension']
        else:
            self.image_folder = None

        obj_label_f = self.config['object_label_folder']
        if obj_label_f is not None:
            obj_label_f += self.folder_postfix
            self.obj_label_folder = os.path.join(self.root_folder, obj_label_f)
            self.obj_label_extension = self.config['object_label_extension']
        else:
            self.obj_label_folder = None

        mat_label_f = self.config['material_label_folder']
        if mat_label_f is not None:
            mat_label_f += self.folder_postfix
            self.mat_label_folder = os.path.join(self.root_folder, mat_label_f)
            self.mat_label_extension = self.config['material_label_extension']
        else:
            self.mat_label_folder = None

        calib_f = self.config.get('calibration_folder')
        if calib_f is not None:
            calib_f += self.folder_postfix
            self.calibration_folder = os.path.join(self.root_folder, calib_f)
            self.calibration_extension = self.config.get('calibration_extension')
        else:
            self.calibration_folder = None

        depth_f = self.config.get('depth_folder')
        if depth_f is not None:
            depth_f += self.folder_postfix
            self.depth_folder = os.path.join(self.root_folder, depth_f)
            self.depth_extension = self.config.get('depth_extension')
        else:
            self.depth_folder = None



        self.train_filenames = self.config['train_images']
        self.test_filenames = self.config['test_images']
        self.dataset = self.config['dataset_name']

        self.color_coding = { 'mat' : dataset_utils.LabelConversion(self.config['material_color_coding']),
                              'obj' : dataset_utils.LabelConversion(self.config['object_color_coding'])}

        self.class_count = {k : self.color_coding[k].class_count for k in self.color_coding.keys()}
        self.class_names = {k : self.color_coding[k].class_names for k in self.color_coding.keys()}


    def label_to_rgb(self, image, type):
        return self.color_coding[type].label_to_rgb(image)

    def rgb_to_label(self, image, type):
        return self.color_coding[type].rgb_to_label(image)


    def get_data(self, data_type, color_images=True, mat_label_images=True, obj_label_images=True, calibrations=False, depth=False):
        file_list = []
        for t in data_type:
            list_type = t + '_images'
            if list_type in self.config:
                file_list += self.config[list_type]
            else:
                raise Exception('The config does not contain a list for the entry: \'{0}_images\' \nConfig file located at: {1}'.format(t, self.config_filename))

        return_list = []
        if color_images:
            images = []
            for fn in tqdm(file_list):
                i_n = os.path.join(self.image_folder, fn+self.image_extension)
                images.append(self.load_color(i_n))
            return_list.append(images)

        if mat_label_images:
            mat_labels = []
            for fn in tqdm(file_list):
                mat_l_n = os.path.join(self.mat_label_folder, fn+self.mat_label_extension)
                mat_labels.append(self.load_labels(mat_l_n, 'mat'))
            return_list.append(mat_labels)

        if obj_label_images:
            obj_labels = []
            for fn in tqdm(file_list):
                obj_l_n = os.path.join(self.obj_label_folder, fn+self.obj_label_extension)
                obj_labels.append(self.load_labels(obj_l_n, 'obj'))
            return_list.append(obj_labels)

        if calibrations:
            calibration_data = []
            for fn in tqdm(file_list):
                c_n = os.path.join(self.calibration_folder, fn+self.calibration_extension)
                calibration_data.append(self.load_calibration(c_n))
            return_list.append(calibration_data)

        if depth:
            depth_data = []
            for fn in tqdm(file_list):
                d_n = os.path.join(self.depth_folder, fn+self.depth_extension)
                depth_data.append(self.load_depth(d_n))
            return_list.append(depth_data)

        if len(return_list) == 1:
            return return_list[0]
        else:
            return return_list

    def load_color(self, file_name):
        return cv2.imread(file_name)[:,:,::-1] # flip bgr to rgb

    def load_labels(self, file_name, type):
        rgb = cv2.imread(file_name)[:,:,::-1]
        return self.rgb_to_label(rgb, type)

    def load_calibration(self, file_name):
        with open(file_name) as calib_file:
            return json.load(calib_file)

    def load_depth(self, file_name):
        d = cv2.imread(file_name, cv2.CV_LOAD_IMAGE_UNCHANGED)
        if d.dtype == np.uint16:
            d = d.astype(np.float32)/256.
        return d