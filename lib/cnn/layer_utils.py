from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class sequential(object):
    def __init__(self, *args):
        """
        Sequential Object to serialize the NN layers
        Please read this code block and understand how it works
        """
        self.params = {}
        self.grads = {}
        self.layers = []
        self.paramName2Indices = {}
        self.layer_names = {}

        # process the parameters layer by layer
        for layer_cnt, layer in enumerate(args):
            for n, v in layer.params.items():
                self.params[n] = v
                self.paramName2Indices[n] = layer_cnt
            for n, v in layer.grads.items():
                self.grads[n] = v
            if layer.name in self.layer_names:
                raise ValueError("Existing name {}!".format(layer.name))
            self.layer_names[layer.name] = True
            self.layers.append(layer)

    def assign(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].params[name] = val

    def assign_grads(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].grads[name] = val

    def get_params(self, name):
        # return the parameters by name
        return self.params[name]

    def get_grads(self, name):
        # return the gradients by name
        return self.grads[name]

    def gather_params(self):
        """
        Collect the parameters of every submodules
        """
        for layer in self.layers:
            for n, v in layer.params.items():
                self.params[n] = v

    def gather_grads(self):
        """
        Collect the gradients of every submodules
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                self.grads[n] = v

    def apply_l1_regularization(self, lam):
        """
        Gather gradients for L1 regularization to every submodule
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                ########### TODO #############
                param = self.params[n]
                grad = (param > 0).astype(np.float32) - (param < 0).astype(np.float32)
                self.grads[n] += lam * grad
                ########### END  #############

    def apply_l2_regularization(self, lam):
        """
        Gather gradients for L2 regularization to every submodule
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                ########### TODO #############
                self.grads[n] += lam * self.params[n]
                ########### END  #############


    def load(self, pretrained):
        """
        Load a pretrained model by names
        """
        for layer in self.layers:
            if not hasattr(layer, "params"):
                continue
            for n, v in layer.params.items():
                if n in pretrained.keys():
                    layer.params[n] = pretrained[n].copy()
                    print ("Loading Params: {} Shape: {}".format(n, layer.params[n].shape))

class ConvLayer2D(object):
    def __init__(self, input_channels, kernel_size, number_filters, 
                stride=1, padding=0, init_scale=.02, name="conv"):
        
        self.name = name
        self.w_name = name + "_w"
        self.b_name = name + "_b"

        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.number_filters = number_filters
        self.stride = stride
        self.padding = padding

        self.params = {}
        self.grads = {}
        self.params[self.w_name] = init_scale * np.random.randn(kernel_size, kernel_size, 
                                                                input_channels, number_filters)
        self.params[self.b_name] = np.zeros(number_filters)
        self.grads[self.w_name] = None
        self.grads[self.b_name] = None
        self.meta = None
    
    def get_output_size(self, input_size):
        '''
        :param input_size - 4-D shape of input image tensor (batch_size, in_height, in_width, in_channels)
        :output a 4-D shape of the output after passing through this layer (batch_size, out_height, out_width, out_channels)
        '''
        output_shape = [None, None, None, None]
        #############################################################################
        # TODO: Implement the calculation to find the output size given the         #
        # parameters of this convolutional layer.                                   #
        #############################################################################
        for i in range(len(output_shape)):
            if i != 0:
                #print(input)
                new_output = ( (input_size[i] + (2 * self.padding) - self.kernel_size ) / self.stride) + 1
                output_shape[i] = int(new_output)
            else:
                output_shape[i] = input_size[0]
        output_shape[-1] = self.number_filters

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return output_shape

    def forward(self, img):
        output = None
        assert len(img.shape) == 4, "expected batch of images, but received shape {}".format(img.shape)

        output_shape = self.get_output_size(img.shape)
        _ , input_height, input_width, _ = img.shape
        _, output_height, output_width, _ = output_shape

        #############################################################################
        # TODO: Implement the forward pass of a single convolutional layer.       #
        # Store the results in the variable "output" provided above.                #
        #############################################################################
        
        stride = self.stride
        input_channels = self.input_channels
        img_pad = np.pad(img, ((0,), (self.padding,), (self.padding,), (0,)), mode='constant')

        # Compute strides for sliding window view
        new_shape = ( img.shape[0], output_height, output_width, self.kernel_size, self.kernel_size, input_channels)
        new_strides = (
            img_pad.strides[0],
            stride * img_pad.strides[1],
            stride * img_pad.strides[2],
            img_pad.strides[1],
            img_pad.strides[2],
            img_pad.strides[3]
        )

        # Create sliding window view of padded input image
        img_as_kern = np.lib.stride_tricks.as_strided(img_pad, shape=new_shape, strides=new_strides)
        img_as_kern = img_as_kern.reshape(img.shape[0], output_height, output_width, self.kernel_size * self.kernel_size * input_channels)

        # Reshape kernel
        kernel = self.params[self.w_name].reshape(-1, self.number_filters)
        output = img_as_kern @ kernel 
        output += self.params[self.b_name]
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = img
        
        return output


    def backward(self, dprev):
        img = self.meta
        if img is None:
            raise ValueError("No forward function called before for this module!")

        dimg, self.grads[self.w_name], self.grads[self.b_name] = None, None, None
        
        #############################################################################
        # TODO: Implement the backward pass of a single convolutional layer.        #
        # Store the computed gradients wrt weights and biases in self.grads with    #
        # corresponding name.                                                       #
        # Store the output gradients in the variable dimg provided above.           #
        #############################################################################
        
        img_pad = np.pad(img, ((0,), (self.padding,), (self.padding,), (0,)), mode='constant')
        w_shape = self.params[self.w_name].shape
        dw = np.zeros((w_shape[3] , w_shape[0], w_shape[1], w_shape[2]))
        self.grads[self.b_name] = np.zeros(self.params[self.b_name].shape)
        dimg_pad = np.zeros_like(img_pad)
        dimg = np.zeros_like(img)

        for batch in range(dprev.shape[0]):
            for pixel_h in range(dprev.shape[1]):
                for pixel_w in range (dprev.shape[2]):
                    # cropped_img = img_pad[:, pixel_h * self.stride: self.kernel_size +  (pixel_h * self.stride) , pixel_w * self.stride: (pixel_w * self.stride) + self.kernel_size, :]
                    # print(cropped_img.shape)
                    
                    window = img_pad[batch, pixel_h*self.stride:pixel_h*self.stride+self.kernel_size, pixel_w*self.stride:pixel_w*self.stride+self.kernel_size, :]
                    dw += dprev[batch][pixel_h][pixel_w].reshape((-1,1,1,1)) * window 

                    dimg_pad[batch, pixel_h*self.stride:pixel_h*self.stride+self.kernel_size,
                            pixel_w*self.stride:pixel_w*self.stride+self.kernel_size, :] += \
                        np.sum(dprev[batch, pixel_h, pixel_w, :].reshape(1, 1, 1, -1) *
                            self.params[self.w_name], axis=3)

        dimg = dimg_pad[:, self.padding:img.shape[1]+self.padding, self.padding:img.shape[2]+self.padding, :]   
        self.grads[self.w_name] = dw.transpose(1,2,3,0)
        self.grads[self.b_name] = np.sum(dprev, axis=(0, 1, 2))
        
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        self.meta = None
        return dimg


class MaxPoolingLayer(object):
    def __init__(self, pool_size, stride, name):
        self.name = name
        self.pool_size = pool_size
        self.stride = stride
        self.params = {}
        self.grads = {}
        self.meta = None

    def forward(self, img):
        output = None
        assert len(img.shape) == 4, "expected batch of images, but received shape {}".format(img.shape)
        
        #############################################################################
        # TODO: Implement the forward pass of a single maxpooling layer.            #
        # Store your results in the variable "output" provided above.               #
        #############################################################################
        pool_s = self.pool_size
        out_width = int ( np.floor( (img.shape[2] - self.pool_size) / 2) + 1 )
        out_height = int ( np.floor( (img.shape[1] - self.pool_size) / 2) + 1 )
        output = np.zeros((img.shape[0], out_height, out_width,  img.shape[-1]))
        for p_h in range(out_height):
            for p_w in range(out_height): 
                pool_window = img[:,
                                   p_h * self.stride: pool_s + p_h * self.stride ,   
                                   p_w * self.stride: pool_s + p_w * self.stride, 
                                   :]
                output[:, p_h, p_w, :] = np.max(pool_window, axis=(1,2))
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = img

        return output

    def backward(self, dprev):
        img = self.meta

        dimg = np.zeros_like(img)
        _, h_out, w_out, _ = dprev.shape
        h_pool, w_pool = self.pool_size,self.pool_size

        #############################################################################
        # TODO: Implement the backward pass of a single maxpool layer.              #
        # Store the computed gradients in self.grads with corresponding name.       #
        # Store the output gradients in the variable dimg provided above.           #
        #############################################################################
        d_tensor =  np.zeros_like(img)
        #print(d_tensor.shape)

        for batch in range(img.shape[0]):
            for p_h in range(h_out):
                for p_w in range(w_out):
                    for channel in range(img.shape[-1]):
                        pool_window = img[batch,
                                        p_h * self.stride  :  h_pool +  p_h * self.stride,
                                        p_w * self.stride  :  w_pool +  p_w * self.stride,
                                        channel]
                        max_in = np.unravel_index(np.argmax(pool_window), pool_window.shape)
                        dimg[batch, p_h * self.stride + max_in[0], p_w * self.stride + max_in[1], channel] += dprev[batch, p_h, p_w, channel]
        
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return dimg
