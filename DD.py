import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

import time
import random
import math
import copy

from PIL import Image

from keras import backend
from keras.models import Model
from keras.applications.vgg19 import VGG19

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
from scipy.ndimage.filters import gaussian_filter




height = 512
width = 512

content_image = Image.open('hugo.jpg')
content_image = content_image.resize((height, width))
content_array = np.asarray(content_image, dtype='float32')
content_array = np.expand_dims(content_array, axis=0)

style_image = Image.open('wave.jpg')
style_image = style_image.resize((height, width))
style_array = np.asarray(style_image, dtype='float32')
style_array = np.expand_dims(style_array, axis=0)

def from_RGB_to_BGR(x2):
    x = copy.deepcopy(x2)
    x[:, :, :, 0] -= 103.939
    x[:, :, :, 1] -= 116.779
    x[:, :, :, 2] -= 123.68
    return x[:, :, :, ::-1]

content_array = from_RGB_to_BGR(content_array)
style_array = from_RGB_to_BGR(style_array)

content_image = backend.variable(content_array)
style_image = backend.variable(style_array)
combination_image = backend.placeholder((1, height, width, 3))

input_tensor = backend.concatenate([content_image,
                                    style_image,
                                    combination_image], axis=0)

model = VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)

'''
The loss function we want to minimise can be decomposed into three distinct parts: the content loss, the style loss and the total variation loss
'''
content_weight = 0.025
style_weight = 5.0
total_variation_weight = 1.0

layers = dict([(layer.name, layer.output) for layer in model.layers])

# how different in content two images are from one another
loss = backend.variable(0.)


'''The content loss'''
# The content loss is the (scaled, squared) Euclidean distance between feature representations of the content and combination images
def content_loss(content, combination):
    return backend.sum(backend.square(combination - content))


layer_features = layers['block2_conv2']
feature_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3',
                  'block4_conv3', 'block5_conv3']
content_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]

loss += content_weight * content_loss(content_image_features, combination_features)


'''The style loss'''
# The style loss is then the(scaled, squared) Frobenius norm of the difference between the Gram matrices of the style and combination images.

# The Gram matrix can be computed efficiently by reshaping the feature spaces suitably and taking an outer product
def gram_matrix(x):
    features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
    gram = backend.dot(features, backend.transpose(features))
    return gram

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = height * width
    return backend.sum(backend.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

for layer_name in feature_layers:
    layer_features = layers[layer_name]
    style_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_features, combination_features)
    loss += (style_weight / len(feature_layers)) * sl


'''The total variation loss'''
# a regularisation term that encourages spatial smoothness
# Total variation loss to ensure the image is smooth and continuous throughout
def total_variation_loss(x):
    a = backend.square(x[:, :height-1, :width-1, :] - x[:, 1:, :width-1, :])
    b = backend.square(x[:, :height-1, :width-1, :] - x[:, :height-1, 1:, :])
    return backend.sum(backend.pow(a + b, 1.25))

loss += total_variation_weight * total_variation_loss(combination_image)


'''Optimisation problem'''
# define gradients of the total loss relative to the combination image, and use these gradients to iteratively improve upon our combination image to minimise the loss
grads = backend.gradients(loss, combination_image)

outputs = [loss]
outputs += grads
f_outputs = backend.function([combination_image], outputs)

def eval_loss_and_grads(x):
    x = x.reshape((1, height, width, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    grad_values = outs[1].flatten().astype('float64')
    return loss_value, grad_values

class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


evaluator = Evaluator()


x = np.random.uniform(0, 255, (1, height, width, 3)) - 128.0

for i in range(20):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time - start_time))

    try:
        x1 = copy.deepcopy(x)
        x1 = x1.reshape((height, width, 3))
        # Convert back from BGR to RGB to display the image
        x1 = x1[:, :, ::-1]
        x1[:, :, 0] += 103.939
        x1[:, :, 1] += 116.779
        x1[:, :, 2] += 123.68
        x1 = np.clip(x1, 0, 255).astype('uint8')
        img_final = Image.fromarray(x1)
        img_final.save('result' + str(i) + '.bmp')
        img_final.show()
    except:
        pass

