import scipy.ndimage
import numpy as np


class flip_augmentation(object):
    def __init__(self, chance=0.5, image_x_axis=-1, semantics_x_axis=-1):
        '''
        The standard parameters will flip the image along the x axis, 
        assuming a 'bc01' format for the image and a 'b01' format for
        the semantic image.
        '''
        self.chance=chance
        self.image_x_axis=image_x_axis
        self.semantics_x_axis=semantics_x_axis


    def apply(self, image, semantic_image):
        if np.random.uniform() < self.chance:
            return (np.swapaxes(np.swapaxes(image, 0, self.image_x_axis)[::-1], 0, self.image_x_axis),
                    np.swapaxes(np.swapaxes(semantic_image, 0, self.semantics_x_axis)[::-1], 0, self.semantics_x_axis))
        else:
            return image, semantic_image



class scale_augmentation(object):
    def __init__(self, max_x=1.2, max_y=1.2, image_axis_y_x=[-2, -1], semantic_axis_y_x=[-2, -1]):
        '''
        max_x/max_y speficy the maximum scaling factor. The minimum corresponds to 1.0/max.
        the axis specify which axis to use as y and x axis.
        '''
        self.x_min = np.log(1./max_x)/np.log(2.0)
        self.x_max = np.log(max_x)/np.log(2.0)
        self.y_min = np.log(1./max_y)/np.log(2.0)
        self.y_max = np.log(max_y)/np.log(2.0)
        self.image_axis_y_x = image_axis_y_x
        self.semantic_axis_y_x = semantic_axis_y_x


    def apply(self, image, semantic_image):
        x = np.power(2.0,np.random.uniform(low=self.x_min, high=self.x_max))
        y = np.power(2.0,np.random.uniform(low=self.y_min, high=self.y_max))
        zoom_im = np.ones(len(image.shape))
        zoom_im[self.image_axis_y_x] = y, x
        im = scipy.ndimage.interpolation.zoom(image, zoom=zoom_im, order=0)
        zoom_ta = np.ones(len(semantic_image.shape))
        zoom_ta[self.semantic_axis_y_x] = y, x
        ta = scipy.ndimage.interpolation.zoom(semantic_image, zoom=zoom_ta, order=0)
        return im, ta

class pca_color_augmentation(object):
    def __init__(self, sigma=0.1, color_axis=1, color_channels=[0,1,2]):
        '''
        color_axis represents the color_axis in the images when the augmentation is applied.
        This is not considered for the training step. The color_channels parameter is used
        in both cases though and should be consistent!
        '''
        self.sigma=sigma
        self.color_axis=color_axis
        self.color_channels=color_channels



    def train(self, color_images, color_axis):
        '''
        Expects a list of color images, each having 3 dimensions. The color axis needs to be specified.
        '''
        self.d = len(self.color_channels)
        pixel_counts = [np.prod(np.swapaxes(im,color_axis, -1).shape[0:2]) for im in color_images]
        pixels=np.zeros((np.sum(pixel_counts), self.d), dtype=np.float32)
        count = 0
        for im, current_count in zip(color_images, pixel_counts):
            pixels[count:(count+current_count)] = np.swapaxes(im,color_axis, -1).reshape(current_count, -1)[:, self.color_channels]
            count += current_count

        self.data_mean = np.mean(pixels, 0)
        pixels = pixels - self.data_mean
        self.covariance = np.dot(pixels.T, pixels)/pixels.shape[0]
        
        self.u,s,v = np.linalg.svd(self.covariance)
        self.ev = np.sqrt(s)


    def apply(self, image, semantic_image):
        color_noise =np.dot(self.u, np.random.normal(0.0, self.sigma, self.d)*self.ev)
        augmented = np.swapaxes(image, self.color_axis, -1).astype(np.float32).copy()
        augmented[..., self.color_channels] += color_noise
        augmented = np.swapaxes(augmented, self.color_axis, -1)
               
        return augmented, semantic_image