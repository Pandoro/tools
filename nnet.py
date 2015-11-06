import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import theano.sandbox.cuda.dnn as dnn
import theano.sandbox.cuda.basic_ops as basic_ops
import numpy as np

class LogisticRegression(object):
    def __init__(self, input, n_in=None, n_out=None, W=None, b=None):
        # init weights and bias
        if W is None:
            self.W = theano.shared(
                value=np.zeros(
                    (n_in, n_out),
                    dtype=theano.config.floatX
                ),
                name='W',
                borrow=True
            )
        else:
            self.W = W

        if b is None:
            self.b = theano.shared(
                value=np.zeros(
                    (n_out,),
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )
        else:
            self.b = b

        # softmax output
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]
        self.input = input


    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])


    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()



class Regression(object):
    def __init__(self, input, n_in=None, n_out=None, W=None, b=None):
        # init weights and bias
        if W is None:
            self.W = theano.shared(
                value=np.zeros(
                    (n_in, n_out),
                    dtype=theano.config.floatX
                ),
                name='W',
                borrow=True
            )
        else:
            self.W = W

        if b is None:
            self.b = theano.shared(
                value=np.zeros(
                    (n_out,),
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )
        else:
            self.b = b

        self.output = T.dot(input, self.W)+ self.b
        self.params = [self.W, self.b]
        self.input = input


    def quadratic_error(self, y):
        #reshape needed as y is a vector but apparently we need inputs 2 dimensions as in (n,1)
        return T.mean((self.output - y.reshape((-1, 1)))**2) 


    def conditional_quadratic_error(self, y, condition):
        #reshape needed as y is a vector but apparently we need inputs 2 dimensions as in (n,1)
        return T.mean(condition.reshape((-1, 1))*((self.output - y.reshape((-1, 1)))**2.0))


    def sign_error(self, y, condition):
        return T.mean(condition.reshape((-1, 1))*T.neq(T.sgn(self.output), T.sgn(y.reshape((-1, 1)))))



class HiddenLayer(object):
    def __init__(self, rng, input, n_in=None, n_out=None, W=None, b=None,
                 activation=T.tanh):
        self.input = input
        #init weight and bias if none are given.
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        # store them
        self.W = W
        self.b = b

        #also store the activation function for twin layers
        self.activation = activation

        #apply activation
        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )

        # parameters of the model
        self.params = [self.W, self.b]



class ConvLayer(object):
    def __init__(self, rng, input, filter_shape=None, image_shape=None,activation=T.tanh,W=None, b=None, init='something', border=None):

        if not image_shape is  None:
            assert image_shape[1] == filter_shape[1]

        #weight and bias init if none are given.
        if init == 'zero':
            if W is None:
                W_values = np.zeros(filter_shape, dtype=theano.config.floatX)
                W = theano.shared(value=W_values, name='W_conv', borrow=True)
        else:
            if W is None:
                fan_in = np.prod(filter_shape[1:])
                fan_out = (filter_shape[0] * np.prod(filter_shape[2:]))
                # initialize weights with random weights
                W_bound = np.sqrt(6. / (fan_in + fan_out))
                W = theano.shared(
                    np.asarray(
                        rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                        dtype=theano.config.floatX
                    ), name='W_conv',
                    borrow=True
                )

        if b is None:
            # the bias is a 1D tensor -- one bias per output feature map
            b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b_conv', borrow=True)

        self.W = W
        self.b = b

        self.filter_shape = W.shape.eval()
        #self.image_shape = image_shape

        #if border == 'same':
        #    border_mode = 'full'

        #conv_out = conv.conv2d(
        #    input=input,
        #    filters=self.W,
        #    filter_shape=filter_shape,
        #    #image_shape=image_shape,
        #    border_mode=border_mode
        #)

        #if border == 'same':
        #    border_l = T.shape(W)[2] - 1  # this is the filter size minus 1
        #    conv_out = conv_out[:,:,border_l:(T.shape(input)[2]+border_l),border_l:(T.shape(input)[3]+border_l)]

        self.border = border
        if border == 'same':
            assert self.filter_shape[2] % 2 == 1 and self.filter_shape[3] % 2 == 1
            self.border_padding = ((self.filter_shape[2]-1)//2, (self.filter_shape[3]-1)//2)
        elif border == 'valid':
            self.border_padding = (0,0)
        else:
            return NotImplementedError()

        conv_out = dnn.dnn_conv(
            img = input,
            kerns = self.W,
            border_mode = self.border_padding
        )

        self.activation = activation

        #adding the bias
        if self.activation is None:
            self.output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        else:
            self.output = self.activation(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        self.input = input



class AveragePoolLayer(object):
    def __init__(self, input, poolsize=(2, 2)):
        self.poolsize=poolsize
        self.input = input
        self.output = downsample.max_pool_2d(
            input=self.input.dimshuffle(0,1,3,2),
            ds=self.poolsize,
            ignore_border=False,
            mode = 'average_exc_pad'
        ).dimshuffle(0,1,3,2) # the two dimshuffles are needed because the last dimensions output cannot be over 512!



class MaxPoolLayer(object):
    def __init__(self, input, poolsize=(2, 2)):
        self.poolsize=poolsize
        self.input = input
        self.output = downsample.max_pool_2d(
            input=self.input.dimshuffle(0,1,3,2),
            ds=self.poolsize,
            ignore_border=False
        ).dimshuffle(0,1,3,2) # the two dimshuffles are needed because the last dimensions output cannot be over 512!

        #self.output = dnn.dnn_pool(
        #    img=self.input,
        #    ws=poolsize,
        #    stride=poolsize,
        #    mode='max'
        #)


class UnPoolLayer(object):
    def __init__(self, input, poolsize=(2, 2), output_shape=None):
        self.poolsize=poolsize
        self.input = input
        
        self.output = input
        if poolsize[0] > 1:
            self.output = T.extra_ops.repeat(self.output, poolsize[0], 2)
        if poolsize[1] > 1:
            self.output = T.extra_ops.repeat(self.output, poolsize[1], 3)
            
        #We might need to crop away parts of the output.
        if output_shape is not None:
            if T.lt(output_shape[2], self.output.shape[2]):
                self.output = self.output[:,:,:output_shape[2],:]
            if T.lt(output_shape[3], self.output.shape[3]):
                self.output = self.output[:,:,:,:output_shape[3]]



class SoftMax(object):
    def __init__(self, input):
        self.p_y_given_x = T.nnet.softmax(input)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.dtype.startswith('int') and y.ndim == 1:
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()



class SoftMax_Spatial(object):
    def __init__(self, input, clipping_epsilon=0):
        self.clipping_epsilon = clipping_epsilon
        self.p_y_given_x_linear = T.clip(T.nnet.softmax(input.swapaxes(0,1).flatten(2).swapaxes(0,1)), self.clipping_epsilon, 1.0)
        self.p_y_given_x = self.p_y_given_x_linear.swapaxes(0,1).reshape((input.shape[1],) +  (input.shape[0],) + (input.shape[2],) + (input.shape[3],)).swapaxes(0,1)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

    def weighted_negative_log_likelihood(self, y, weights):
        yflat = y.flatten()
        weights_flat=weights.flatten()
        return -T.mean(weights_flat*T.log(self.p_y_given_x_linear)[T.arange(yflat.shape[0]), yflat])

    def negative_log_likelihood(self, y):
        yflat = y.flatten()
        mask = T.neq(yflat, -1)
        #TODO should this really be the mean or should be sum over it and divide by only those that are not void?
        return -T.mean(mask*T.log(self.p_y_given_x_linear)[T.arange(yflat.shape[0]), yflat])

    def log_likelihood_im(self, y):
        yflat = y.flatten()
        mask = T.neq(yflat, -1)
        #TODO should this really be the mean or should be sum over it and divide by only those that are not void?
        return (mask*T.log(self.p_y_given_x_linear)[T.arange(yflat.shape[0]), yflat]).reshape(y.shape)

    def errors(self, y):
        if y.dtype.startswith('int') and y.ndim == 3:
            mask = T.neq(y, -1)
            total = T.sum(mask, dtype='float32')
            return T.sum(T.neq(self.y_pred, y)*mask)/total
        else:
            raise NotImplementedError()

#class SoftMax_Spatial_CUDNN(object):
#    def __init__(self, input):
#        inp = basic_ops.gpu_contiguous(input)
#        self.p_y_given_x = dnn.GpuDnnSoftmax(tensor_format='bc01', algo='log', mode='channel')(inp)
#        self.p_y_given_x_linear = self.p_y_given_x.swapaxes(0,1).flatten(2).swapaxes(0,1)
#        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

#    def negative_log_likelihood(self, y):
#        yflat = y.flatten()
#        return -T.mean(self.p_y_given_x_linear[T.arange(yflat.shape[0]), yflat])

#    def errors(self, y):
#        if y.dtype.startswith('int') and y.ndim == 3:
#            return T.mean(T.neq(self.y_pred, y))
#        else:
#            raise NotImplementedError()


class ConvolutionSoftMaxCombo(object):
    def __init__(self, rng, input, filter_shape, clipping_epsilon=0.0):
        self.conv = ConvLayer(
            rng,
            input=input,
            filter_shape=filter_shape,
            activation=None,
            border='same',
            init='zero'
        )
        self.softmax = SoftMax_Spatial(input=self.conv.output, clipping_epsilon=clipping_epsilon)

        self.p_y_given_x_linear = self.softmax.p_y_given_x_linear
        self.p_y_given_x = self.softmax.p_y_given_x
        self.y_pred = self.softmax.y_pred

        self.params = self.conv.params


    def weighted_negative_log_likelihood(self, y, weights):
        return self.softmax.weighted_negative_log_likelihood(y, weights)

    def negative_log_likelihood(self, y):
        return self.softmax.negative_log_likelihood(y)

    def log_likelihood_im(self, y):
        return self.softmax.log_likelihood_im(y)

    def errors(self, y):
        return self.softmax.errors(y)



class Optimizer(object):
    def __init__(self,learning_rate, params, cost, input, output):
        self.params = params

        self.cost = cost

        self.grads = T.grad(self.cost, self.params)
        self.grad_storage = [theano.shared(np.zeros_like(p.get_value(borrow=True))) for p in self.params]

        self.update_grads = [
            (storage_i, grad_i)
            for storage_i, grad_i in zip(self.grad_storage, self.grads)
        ]

        self.update_params = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(self.params, self.grad_storage)
        ]


        inp = []
        if type(input) == list:
            inp += input
        else:
            inp.append(input)

        if type(output) == list:
            inp += output
        else:
            inp.append(output)
        self.get_gradients = theano.function(
            inputs=inp,
            outputs=self.cost,
            updates=self.update_grads
        )

        self.update_model = theano.function(
            inputs=[],
            updates=self.update_params
        )