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
                ######## TODO ########

                # this layer.grads[n] += np.sum(np.abs(v)) * lam
                # or this layer.grads[n] += np.sign(layer.params[n]) * lam
               
                layer.grads[n] += lam * np.sign(v)
                ######## END  ########
    
    def apply_l2_regularization(self, lam):
        """
        Gather gradients for L2 regularization to every submodule
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                ######## TODO ########
                
                layer.grads[n] += layer.params[n] * lam
                ######## END  ########


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


class flatten(object):
    def __init__(self, name="flatten"):
        """
        - name: the name of current layer
        - meta: to store the forward pass activations for computing backpropagation
        Note: params and grads should be just empty dicts here, do not update them
        """
        self.name = name
        self.params = {}
        self.grads = {}
        self.meta = None

    def forward(self, feat):
        output = None
        #############################################################################
        # TODO: Implement the forward pass of a flatten layer.                      #
        # You need to reshape (flatten) the input features.                         #
        # Store the results in the variable self.meta provided above.               #
        #############################################################################
        output = feat.reshape(feat.shape[0], -1)
        #print(output.shape)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = feat
        return output

    def backward(self, dprev):
        feat = self.meta
        if feat is None:
            raise ValueError("No forward function called before for this module!")
        dfeat = None
        #############################################################################
        # TODO: Implement the backward pass of a flatten layer.                     #
        # You need to reshape (flatten) the input gradients and return.             #
        # Store the results in the variable dfeat provided above.                   #
        #############################################################################
        dfeat = dprev.reshape(feat.shape)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = None
        return dfeat


class fc(object):
    def __init__(self, input_dim, output_dim, init_scale=0.002, name="fc"):
        """
        In forward pass, please use self.params for the weights and biases for this layer
        In backward pass, store the computed gradients to self.grads
        - name: the name of current layer
        - input_dim: input dimension
        - output_dim: output dimension
        - meta: to store the forward pass activations for computing backpropagation
        """
        self.name = name
        self.w_name = name + "_w"
        self.b_name = name + "_b"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.params = {}
        self.grads = {}
        self.params[self.w_name] = init_scale * np.random.randn(input_dim, output_dim)
        self.params[self.b_name] = np.zeros(output_dim)
        self.grads[self.w_name] = None
        self.grads[self.b_name] = None
        self.meta = None

    def forward(self, feat):
        output = None
        assert len(feat.shape) == 2 and feat.shape[-1] == self.input_dim, \
            "But got {} and {}".format(feat.shape, self.input_dim)
        #############################################################################
        # TODO: Implement the forward pass of a single fully connected layer.       #
        # Store the results in the variable output provided above.                  #
        #############################################################################
        output = feat @ self.params[self.w_name] + self.params[self.b_name]
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = feat
        return output

    def backward(self, dprev):
        feat = self.meta
        if feat is None:
            raise ValueError("No forward function called before for this module!")
        dfeat, self.grads[self.w_name], self.grads[self.b_name] = None, None, None
        assert len(feat.shape) == 2 and feat.shape[-1] == self.input_dim, \
            "But got {} and {}".format(feat.shape, self.input_dim)
        assert len(dprev.shape) == 2 and dprev.shape[-1] == self.output_dim, \
            "But got {} and {}".format(dprev.shape, self.output_dim)
        #############################################################################
        # TODO: Implement the backward pass of a single fully connected layer.      #
        # Store the computed gradients wrt weights and biases in self.grads with    #
        # corresponding name.                                                       #
        # Store the output gradients in the variable dfeat provided above.          #
        #############################################################################
        """
        NEEDS 
            input into that layer
            deriv of wieghts
            deriv of bias v
            store in grad[w]
            grad[b]
            output = 
        """

        self.grads[self.b_name] = dprev.sum(0)
        #print(dprev.shape, feat.shape)
        self.grads[self.w_name] = feat.T @ dprev
        #print(dprev.shape, self.params[self.w_name].shape)
        dfeat = dprev @ (self.params[self.w_name]).T
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = None
        return dfeat

class gelu(object):
    def __init__(self, name="gelu"):
        """
        - name: the name of current layer
        - meta:  to store the forward pass activations for computing backpropagation
        Notes: params and grads should be just empty dicts here, do not update them
        """
        self.name = name 
        self.params = {}
        self.grads = {}
        self.meta = None 
    
    def forward(self, feat):
        output = None
        #############################################################################
        # TODO: Implement the forward pass of GeLU                                  #
        # Store the results in the variable output provided above.                  #
        #############################################################################
        temp_div = (2 / np.pi)**0.5  # scalar
        intermed_mat = feat + (0.044715 * (feat ** 3))
        # print(intermed_mat.shape)
        final_mat = temp_div * intermed_mat
        final_mat = 1 + np.tanh(final_mat)
        #print(feat.shape, final_mat.shape)

        output = (0.5 * feat) * final_mat
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = feat
        return output
    
    def backward(self, dprev):
        """ You can use the approximate gradient for GeLU activations """
        feat = self.meta
        if feat is None:
            raise ValueError("No forward function called before for this module!")
        dfeat = None
        #############################################################################
        # TODO: Implement the backward pass of GeLU                                 #
        # Store the output gradients in the variable dfeat provided above.          #
        #############################################################################
        x_cubed = feat ** 3
        inner_term = (0.0356774 * x_cubed) + (0.797885 * feat)
        d_gelu = (0.5 * np.tanh(inner_term)) + 0.5 + ((0.0535161 *
                                                       x_cubed) + (0.398942 * feat)) * (np.cosh(inner_term) ** -2)
        dfeat = d_gelu * dprev
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = None
        return dfeat



class dropout(object):
    def __init__(self, keep_prob, seed=None, name="dropout"):
        """
        - name: the name of current layer
        - keep_prob: probability that each element is kept.
        - meta: to store the forward pass activations for computing backpropagation
        - kept: the mask for dropping out the neurons
        - is_training: dropout behaves differently during training and testing, use
                       this to indicate which phase is the current one
        - rng: numpy random number generator using the given seed
        Note: params and grads should be just empty dicts here, do not update them
        """
        self.name = name
        self.params = {}
        self.grads = {}
        self.keep_prob = keep_prob
        self.meta = None
        self.kept = None
        self.is_training = False
        self.rng = np.random.RandomState(seed)
        assert keep_prob >= 0 and keep_prob <= 1, "Keep Prob = {} is not within [0, 1]".format(keep_prob)

    def forward(self, feat, is_training=True, seed=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        kept = None
        output = None
        #############################################################################
        # TODO: Implement the forward pass of Dropout.                              #
        # Remember if the keep_prob = 0, there is no dropout.                       #
        # Use self.rng to generate random numbers.                                  #
        # During training, need to scale values with (1 / keep_prob).               #
        # Store the mask in the variable kept provided above.                       #
        # Store the results in the variable output provided above.                  #
        #############################################################################
        mask = np.ones((feat.shape[1], 1))
        if is_training and self.keep_prob != 0:
            mask = self.rng.random((feat.shape[1], 1))
            mask = np.ma.masked_where(self.keep_prob > mask, mask)
            mask = np.where(np.ma.is_masked(mask), mask * 0, mask * 1)
            #                            dropout coeff     *   makes it 1
            mask = np.where(mask != 0, (1/self.keep_prob)
                            * (mask + (1 - mask)), mask * 1)
            output = feat * mask.T
        else:
            output = feat

        kept = mask
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.kept = kept
        self.is_training = is_training
        self.meta = feat
        return output

    def backward(self, dprev):
        feat = self.meta
        dfeat = None
        if feat is None:
            raise ValueError("No forward function called before for this module!")
        #############################################################################
        # TODO: Implement the backward pass of Dropout                              #
        # Select gradients only from selected activations.                          #
        # Store the output gradients in the variable dfeat provided above.          #
        #############################################################################
        if self.keep_prob != 0:
            dfeat = dprev * (self.kept).T
        else:
            dfeat = dprev * self.kept.T
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.is_training = False
        self.meta = None
        return dfeat


class cross_entropy(object):
    def __init__(self, size_average=True):
        """
        - size_average: if dividing by the batch size or not
        - logit: intermediate variables to store the scores
        - label: Ground truth label for classification task
        """
        self.size_average = size_average
        self.logit = None
        self.label = None

    def forward(self, feat, label):
        logit = softmax(feat)
        loss = None
        #############################################################################
        # TODO: Implement the forward pass of an CE Loss                            #
        # Store the loss in the variable loss provided above.                       #
        #############################################################################
        N = label.shape[0]

        # print(N)
        # print(feat)
        # print(np.log(logit).shape)
        #print("logit second index size: ", logit.shape[1])
        # print(label)

        one_hot = np.zeros((label.size, logit.shape[1]))
        one_hot[np.arange(label.size), label] = 1
        #print("onehot.shape: ", one_hot.shape, "logit: ", logit.shape)
        loss = - (1/N) * np.sum(one_hot * np.log(logit))
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.logit = logit
        self.label = label
        return loss

    def backward(self):
        logit = self.logit
        label = self.label
        if logit is None:
            raise ValueError("No forward function called before for this module!")
        dlogit = None
        #############################################################################
        # TODO: Implement the backward pass of an CE Loss                           #
        # Store the output gradients in the variable dlogit provided above.         #
        #############################################################################
        """
        temp_label = np.zeros((label.size, label.max() + 1))
        temp_label[np.arange(label.size), label] = 1

        term_one = np.divide(logit,temp_label)
        term_two = np.divide((1 - logit), (1 - temp_label))
        dlogit = np.add(term_one, term_two)
        """

        logit[range(label.shape[0]), label] -= 1
        dlogit = logit / label.shape[0]
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.logit = None
        self.label = None
        return dlogit


def softmax(feat):
    scores = None

    #############################################################################
    # TODO: Implement the forward pass of a softmax function                    #
    # Return softmax values over the last dimension of feat.                    #
    #############################################################################
    #print(( np.sum(np.exp(feat), axis=1)).shape)
    """temp = ( np.sum(np.exp(feat), axis=1))
    print(temp.shape)
    temp2 = temp.reshape(-1, 1)
    print(temp2.shape)
    scores = np.exp(feat) / temp2"""

    scores = (np.exp(feat)/(np.exp(feat).sum(1)).reshape(-1, 1))
    #print("THIS WORKS")
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return scores

def reset_seed(seed):
    import random
    np.random.seed(seed)
    random.seed(seed)
