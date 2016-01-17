import numpy as np

class LabelConversion(object):
    def __init__(self, conversion_data, void_name = 'Void'):

        #I will force you to have a void label! :D
        if not void_name in [l['name'] for l in conversion_data]:
            raise Exception('Please define the \'Void\' label in your color coding.')

        #Parse the l->c and c->l dicts.
        self.label_to_rgb_dict = {x['label'] : x['color']  for x in conversion_data}
        self.rgb_to_label_dict = { np.sum(np.asarray(x['color'])*np.asarray([1,1000, 1000000])) : x['label']  for x in conversion_data}

        #Count the classes
        self.class_count = len(self.label_to_rgb_dict)-1 #the void label does not count.

        #And get teh class names
        self.class_names = {x['label'] : x['name'] for x in conversion_data if x['name'] != void_name}
        self.class_names = [self.class_names[i] for i in range(self.class_count)]


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