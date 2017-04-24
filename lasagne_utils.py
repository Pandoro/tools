import lasagne
import theano.tensor as T

class NormLayer(lasagne.layers.Layer):
    def __init__(self, incoming, epsilon=1e-7,**kwargs):
        super(NormLayer, self).__init__(incoming, **kwargs)
        self.epsilon=epsilon

    def get_output_for(self, input, **kwargs):
        #A bit shitty but for now!
        input = input.reshape((input.shape[0], -1))
        return input/(T.sqrt(T.sum(input**2, 1)+self.epsilon).dimshuffle(0,'x'))

    def get_output_shape_for(self, input_shape):
        return input_shape