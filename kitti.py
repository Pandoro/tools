import json
import os

import sys
sys.path.append('/usr/lib/python2.7/dist-packages') 
import cv2

import numpy as np

from tqdm import *


class Kitti(object):
    def __init__(self, config_filename):
        self.config_filename = config_filename
        with open(config_filename) as config_file:
            self.config = json.load(config_file)

        if self.config['use_relative_paths']:
            self.root_folder = os.path.dirname(config_filename)
        else:
            self.root_folder = ''

        image_f = self.config['image_folder']
        if image_f is not None:
            self.image_folder = os.path.join(self.root_folder, image_f)
            self.image_extension = self.config['image_extension']
        else:
            self.image_folder = None

        label_f = self.config['label_folder']
        if label_f is not None:
            self.label_folder = os.path.join(self.root_folder, label_f)
            self.label_extension = self.config['label_extension']
        else:
            self.label_folder = None

        calib_f = self.config.get('calibration_folder')
        if calib_f is not None:
            self.calibration_folder = os.path.join(self.root_folder, calib_f)
            self.calibration_extension = self.config.get('calibration_extension')
        else:
            self.calibration_folder = None


        disparity_f = self.config.get('disparity_folder')
        if disparity_f is not None:
            self.disparity_folder = os.path.join(self.root_folder, disparity_f)
            self.disparity_extension = self.config.get('disparity_extension')
        else:
            self.disparity_folder = None


        self.result_folder = os.path.join(self.root_folder, self.config['result_folder'])

        self.train_filenames = self.config['train_images']
        self.test_filenames = self.config['test_images']
        self.dataset = self.config['dataset_name']


        if not 'Void' in [l['name'] for l in self.config['color_coding']]:
            raise Exception('Please define the \'Void\' label in your color coding.')
        self.class_count = len(self.config['color_coding'])-1 #the void label does not count.
        self.class_names = [x['name'] for x in self.config['color_coding'] if x['name'] != 'Void']
        self.label_to_rgb_dict = {x['label'] : x['color']  for x in self.config['color_coding']}
        self.rgb_to_label_dict = { np.sum(np.asarray(x['color'])*np.asarray([1,1000, 1000000])) : x['label']  for x in self.config['color_coding']}


    def label_to_rgb(self, image):
        un_labels, idx = np.unique(image, return_inverse=True)
        rgb = np.asarray([self.label_to_rgb_dict[u] for u in un_labels])
        return rgb[idx].reshape(image.shape +(3,)).astype(np.uint8)


    def rgb_to_label(self, image):
        result = np.zeros(image.shape[0:2], dtype=np.int8)
        colors = np.sum(image*[1, 1000, 1000000],2)
        un_colors = np.unique(colors)
        for u in un_colors:
            l = self.rgb_to_label_dict[u]
            result[colors == u] = l
        return result


    def get_data(self, data_type, color_images=True, label_images=True, calibrations=False, disparity=False):
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

        if label_images:
            labels = []
            for fn in tqdm(file_list):
                l_n = os.path.join(self.label_folder, fn+self.label_extension)
                labels.append(self.load_labels(l_n))
            return_list.append(labels)

        if calibrations:
            calibration_data = []
            for fn in tqdm(file_list):
                c_n = os.path.join(self.calibration_folder, fn+self.calibration_extension)
                calibration_data.append(self.load_calibration(c_n))
            return_list.append(calibration_data)

        if disparity:
            disparity_data = []
            for fn in tqdm(file_list):
                d_n = os.path.join(self.disparity_folder, fn+self.disparity_extension)
                disparity_data.append(self.load_disparity(d_n))
            return_list.append(disparity_data)
        return return_list

    def load_color(self, file_name):
        return cv2.imread(file_name)[:,:,::-1] # flip bgr to rgb

    def load_labels(self, file_name):
        rgb = cv2.imread(file_name)[:,:,::-1]
        return self.rgb_to_label(rgb)

    def load_calibration(self, file_name):
        # We load two calibration matrices (3x3) and assumte that the [2,3] of camera 2 is the baseline (in pixels). 
        # The groundtruth seems to use this too, otherwise there should be a y jump of 2 pixels!
        calib = open(file_name, 'r')
        cams_found = []
        for line in calib.readlines():
            if line[0:2] == 'P2':
                cams_found.append('P2')
                calib_cam1_all = Kitti.parse_calibration_line(line[4:])
            if line[0:2] == 'P3':
                cams_found.append('P3')
                calib_cam2_all = Kitti.parse_calibration_line(line[4:])
        calib.close()
        if len(cams_found) != 2:
            raise Exception('Unrecognized calibration file format' )

        calib_cam1 = np.zeros([3,3], dtype=np.float32)
        calib_cam1[0,:] = calib_cam1_all[0:3]
        calib_cam1[1,:] = calib_cam1_all[4:7]
        calib_cam1[2,:] = calib_cam1_all[8:11]
        calib_cam2 = np.zeros([3,3], dtype=np.float32)
        calib_cam2[0,:] = calib_cam2_all[0:3]
        calib_cam2[1,:] = calib_cam2_all[4:7]
        calib_cam2[2,:] = calib_cam2_all[8:11]
        baseline = calib_cam1_all[3]-calib_cam2_all[3]
        return [calib_cam1, calib_cam2, baseline]

    def load_disparity(self, file_name):
        return cv2.imread(file_name, cv2.CV_LOAD_IMAGE_UNCHANGED)

    @staticmethod
    def parse_calibration_line(line):
        out = []
        for f in line.split(' '):
            try:
                val = float(f)
                out.append(val)
            except ValueError:
                pass
        return out
