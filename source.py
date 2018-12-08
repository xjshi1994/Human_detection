# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
import math
import string
import sys
from numpy import arctan2
import random
import math
import glob as gb
import matplotlib.pyplot as plt
from histogram import gradient, magnitude_orientation, hog, visualise_histogram

PREWITT_GX = np.array([[-1,0,1],
                      [-1,0,1],
                      [-1,0,1]])

PREWITT_GY = np.array([[1,1,1],
                      [0,0,0],
                      [-1,-1,-1]])
# 
CELL_ROW = int(8)
CELL_COL = int(8)

img_name = "train_data/train_positive/crop001030c.bmp"
img = cv.imread(img_name, 1)

WINDOW_ROW = 160
WINDOW_COL = 96

# the number of cell in each window

# row: 20
CELL_ROW_PER_WINDOW = int(WINDOW_ROW / CELL_ROW)
# col: 12
CELL_COL_PER_WINDOW = int(WINDOW_COL / CELL_COL)

BLOCK_ROW = 2
BLOCK_COL = 2

# the number of block in each window
# row: 19
BLOCK_ROW_PER_WINDOW = int(CELL_ROW_PER_WINDOW - BLOCK_ROW + 1)
# col: 11
BLOCK_COL_PER_WINDOW = int(CELL_COL_PER_WINDOW - BLOCK_COL + 1)
IMG_ROW = 160
IMG_COL = 96

# step1: convert color image into gray value
def color2gray(img):
    gray = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
    gray_img = gray.astype(np.uint8)
    return gray_img

# step2: 
# do noramlization
def normalize(ndarr):
    result = np.zeros([IMG_ROW, IMG_COL])
    result = np.abs(ndarr)
    max = np.max(result)
    
    if max > 255:
        # take the max value greater than 255, get normalization ratio
        ratio = math.ceil(max / 255)
        result = np.rint(result / ratio)
    return result.astype(np.uint8)

# return if it is undefined area
def isCal(i, j, bound):
    if i >= bound and \
    i < IMG_ROW - bound and \
    j >= bound and \
    j < IMG_COL - bound:
        return True
    return False

# do the convolution
# (a) slice the matrix according to the center
# (b) np.multiply
def conv(i,j,img,kernel):
    kernel_size = kernel.shape[0]
    kernel_bound = int(kernel_size / 2);
    
    up = i - kernel_bound
    down = i + kernel_bound + 1
    left = j - kernel_bound
    right = j + kernel_bound + 1
    
    sliced = img[up:down, left:right]
    return int(np.sum(sliced * kernel))

# Step2: gradient operator
# (a) Prewitt's operator
# (b) do the convolution
def gradient_operator(gimg):
    resultGX = np.zeros([IMG_ROW, IMG_COL])
    resultGY = np.zeros([IMG_ROW, IMG_COL])
    
    kernel_size = PREWITT_GX.shape[0]
    kernel_bound = int(kernel_size / 2);
    
    # ******
    bound = kernel_bound
    count = 0
    for i in range(IMG_ROW):
        for j in range(IMG_COL):
            if isCal(i, j, bound):
                resultGX[i,j] = conv(i,j,gimg, PREWITT_GX)
                resultGY[i,j] = conv(i,j,gimg, PREWITT_GY)
            else:
                resultGX[i,j] = 0
                resultGY[i,j] = 0
    return resultGX, resultGY

# calculate the magnitude
def magnitude(resultGX,resultGY):
    resultMG = np.zeros([IMG_ROW, IMG_COL])
    for i in range(IMG_ROW):
        for j in range(IMG_COL):
            resultMG[i,j] = math.sqrt(math.pow(resultGX[i,j],2)+ \
                                      math.pow(resultGY[i,j],2))
    return resultMG


# ******HOG********

# unsigned the magnitude angle
def get_orientation(Gx, Gy):
    return np.abs((arctan2(Gy, Gx) * 180 / np.pi))

# get histogram for the cell
def get_histogram(magnitude_slice, orientation_slice):
    hist = np.zeros(9,dtype = np.float);
    for i in range(CELL_ROW):
        for j in range(CELL_COL):
            # 10/20 0.5 left 0 right 1
            divide_res = orientation_slice[i,j] / 20
            left_bin_num = (math.floor(divide_res)) % 9
            right_bin_num = (math.ceil(divide_res)) % 9
            
            left_bin_ratio = (orientation_slice[i,j] - left_bin_num * 20) / 20
            right_bin_ratio = 1 - left_bin_ratio
            
            hist[left_bin_num] += magnitude_slice[i,j] * left_bin_ratio
            hist[right_bin_num] += magnitude_slice[i,j] * right_bin_ratio
    return hist

# get the window, and calculate the cell inside, get 20 * 12 * 9
def get_window_cell(window_magnitude, window_orientation):
    window_cell = np.zeros([CELL_ROW_PER_WINDOW,CELL_COL_PER_WINDOW,9], dtype = np.float)
    
    for i in range(CELL_ROW_PER_WINDOW):
        for j in range(CELL_COL_PER_WINDOW):
            up = i * CELL_ROW
            down = up + CELL_ROW
            left = j * CELL_COL
            right = left + CELL_COL
            window_cell[i,j] = get_histogram(window_magnitude[up:down,left:right], \
                                             window_orientation[up:down,left:right])
    return window_cell

# normalize over block
def L2_norm(block):
    norm = np.sqrt(np.sum(np.square(block)))
    if norm == 0:
        return block
    return block / norm

def normalize_over_block(window_cell):
    final_descriptor = np.zeros([BLOCK_ROW_PER_WINDOW,BLOCK_COL_PER_WINDOW,36], dtype = np.float)
    for i in range(BLOCK_ROW_PER_WINDOW):
        for j in range(BLOCK_COL_PER_WINDOW):
            up = i
            down = i + BLOCK_ROW
            left = j
            right = j + BLOCK_COL
            block = window_cell[up:down, left:right].flatten()
            final_descriptor[i,j] = L2_norm(block)
    return final_descriptor.flatten().tolist()

def get_descriptor(img_path):
    image = cv.imread(img_path, 1)
    gray_image = color2gray(image)
    Gx, Gy = gradient_operator(gray_image)
    mag_img = magnitude(Gx, Gy)
    orientation_img = get_orientation(Gx, Gy)
    window_cell = get_window_cell(mag_img, orientation_img)
    final_descriptor = normalize_over_block(window_cell)
    return final_descriptor
    
def get_trainning_set():
    positive_img_path = gb.glob("train_data/train_positive/*.bmp")
    train_sets = []
    count_p = 0
    for path in positive_img_path:
        sub_train_set = []
        train_pos_des = get_descriptor(path)
        sub_train_set.append(train_pos_des)
        print("--{0:d}--".format(count_p))
        print("mean:{0:f}, std_dev{1:f}, path:{2:s}".format(np.mean(np.array(train_pos_des)), np.std(np.array(train_pos_des)), path))
        sub_train_set.append([0.9])
        train_sets.append(sub_train_set)
        count_p += 1
        
    negative_img_path = gb.glob("train_data/train_negative/*.bmp")
    for path in negative_img_path:
        sub_train_set = []
        train_neg_des = get_descriptor(path)
        print("--{0:d}--".format(count_p))
        print("mean:{0:f}, std_dev{1:f}, path:{2:s}".format(np.mean(np.array(train_neg_des)), np.std(np.array(train_neg_des)),path))
        sub_train_set.append(train_neg_des)
        sub_train_set.append([0.1])
        train_sets.append(sub_train_set)
        count_p += 1
        
    return train_sets

# neural network


#
# Shorthand:
#   "pd_" as a variable prefix means "partial derivative"
#   "d_" as a variable prefix means "derivative"
#   "_wrt_" is shorthand for "with respect to"
#   "w_ho" and "w_ih" are the index of weights from hidden to output layer neurons and input to hidden layer neurons respectively
#
# Comment references:
#
# [1] Wikipedia article on Backpropagation
#   http://en.wikipedia.org/wiki/Backpropagation#Finding_the_derivative_of_the_error
# [2] Neural Networks for Machine Learning course on Coursera by Geoffrey Hinton
#   https://class.coursera.org/neuralnets-2012-001/lecture/39
# [3] The Back Propagation Algorithm
#   https://www4.rgu.ac.uk/files/chapter3%20-%20bp.pdf
"""
init:
    num_inputs:
    num_hidden:
    num_outputs:
    hidden_layer_weights:
    output_layer_weights
    
attribute:
    learning rate
    
method:
    feed_forward: inputs->feed_forward, get result sends to ouput layer, return the result
    train(self, training_inputs, training_outputs)
"""
class NeuralNetwork:
    LEARNING_RATE = 0.5

    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights = None, output_layer_weights = None):
        self.num_inputs = num_inputs

        self.hidden_layer = NeuronLayer(num_hidden, 0)
        self.output_layer = NeuronLayer(num_outputs, 1)

        self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)
        self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights)

    def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
        weight_num = 0
        for h in range(len(self.hidden_layer.neurons)):
            for i in range(self.num_inputs):
                if not hidden_layer_weights:
                    self.hidden_layer.neurons[h].weights.append(random.random()*random.randint(-1,1))
                else:
                    self.hidden_layer.neurons[h].weights.append(hidden_layer_weights[weight_num])
                weight_num += 1
        
    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):
        weight_num = 0
        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layer.neurons)):
                if not output_layer_weights:
                    self.output_layer.neurons[o].weights.append(0.01 * random.random()*random.randint(-1,1))
                else:
                    self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                weight_num += 1
    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        print('Hidden Layer')
        #self.hidden_layer.inspect()
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()


    def feed_forward(self, inputs):
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        return self.output_layer.feed_forward(hidden_layer_outputs)

    # Uses online learning, ie updating the weights after each training case
    def test(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)
        print("target", training_outputs, "prediction", self.output_layer.neurons[0].output)
    def train(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)

        # 1. Output neuron deltas
        pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):

            # ∂E/∂zⱼ
            print("deltas", self.output_layer.neurons[o].calculate_pd_error_wrt_total_net_input(training_outputs[o]))
            pd_errors_wrt_output_neuron_total_net_input[o] = self.output_layer.neurons[o].calculate_pd_error_wrt_total_net_input(training_outputs[o])
            
        # 2. Hidden neuron deltas
        pd_errors_wrt_hidden_neuron_total_net_input = [0] * len(self.hidden_layer.neurons)
        for h in range(len(self.hidden_layer.neurons)):

            # We need to calculate the derivative of the error with respect to the output of each hidden layer neuron
            # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
            d_error_wrt_hidden_neuron_output = 0
            for o in range(len(self.output_layer.neurons)):
                d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].weights[h]

            # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
            pd_errors_wrt_hidden_neuron_total_net_input[h] = d_error_wrt_hidden_neuron_output * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_input()
        
        # 3. Update output neuron weights
        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):
                # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
                pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].calculate_pd_total_net_input_wrt_weight(w_ho)
                
                # Δw = α * ∂Eⱼ/∂wᵢ
                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight
                
        # 4. Update hidden neuron weights
        for h in range(len(self.hidden_layer.neurons)):
            for w_ih in range(len(self.hidden_layer.neurons[h].weights)):

                # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
                pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[h] * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_weight(w_ih)
                
                # Δw = α * ∂Eⱼ/∂wᵢ
                self.hidden_layer.neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight

    def calculate_total_error(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].calculate_error(training_outputs[o])
        return total_error

'''
init:
    (a) num_neurons
    (b) choose whether it is hidden or output
    
attributes:
    neurons(list):
    
method:
    feed_forward(self, inputs): set output of every neuron
'''

class NeuronLayer:
    def __init__(self, num_neurons, hidden_0_output_1):
        self.neurons = []
        if hidden_0_output_1 == 0:
            for i in range(num_neurons):
                self.neurons.append(Hidden_Neuron())
        else:
            for i in range(num_neurons):
                self.neurons.append(Output_Neuron())

    def inspect(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])

    def feed_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs

    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs

    
'''
procedure:
    inputs -> |net input| neuron(squash) -> output -> error

attribute:
    weights(list): previous weights
    inputs(list): 
    output(value): 
    
method:
    calculate_output: assign the value to the output
    calculate_total_net_input(self):
    squash: (a)hidden layer: ReLU (b)output layer: sigmoid
    
    calculate_pd_error_wrt_total_net_input(self, target_output): (a) * (b)
    (a) calculate_pd_error_wrt_output(target_output): ∂Error / ∂Output
    (b) calculate_pd_total_net_input_wrt_input(self): ∂Output / ∂Netinput
    (c) calculate_pd_total_net_input_wrt_weight(self, index): ∂Netinput / ∂wi
'''
class Neuron:
    def __init__(self):
        self.weights = []

    def calculate_output(self, inputs):
        self.inputs = inputs
        self.output = self.squash(self.calculate_total_net_input())
        return self.output

    def calculate_total_net_input(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total

    def calculate_pd_error_wrt_total_net_input(self, target_output):
        print("target", target_output, "output",self.output)
        print("Error to output", self.calculate_pd_error_wrt_output(target_output))
        print("derivate of activation function", self.calculate_pd_total_net_input_wrt_input())
        return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_total_net_input_wrt_input();

    def calculate_error(self, target_output):
        return 0.5 * ((target_output - self.output)**2)

    def calculate_pd_error_wrt_output(self, target_output):
        return -(target_output - self.output)

    def calculate_pd_total_net_input_wrt_weight(self, index):
        return self.inputs[index]

class Hidden_Neuron(Neuron):
    def squash(self, total_net_input):
        return max(0, total_net_input)
    
    def calculate_pd_total_net_input_wrt_input(self):
        if self.output > 0:
            return self.output
        else:
            return 0
    
class Output_Neuron(Neuron):
    def squash(self, total_net_input):
        return 1 / (1 + math.exp(-total_net_input))
    
    def calculate_pd_total_net_input_wrt_input(self):
        return self.output * (1 - self.output)
# main
training_sets = get_trainning_set()
# test_pos_des = get_descriptor("train_data/test_positive/crop001008b.bmp")
# test_neg_des = get_descriptor("train_data/train_negative/00000090a_cut.bmp")
# test_set1 = [[test_pos_des,[1]],[test_neg_des,[0]]]
nn = NeuralNetwork(len(training_sets[0][0]), 100, len(training_sets[0][1]))
for i in range(10000):
    training_inputs, training_outputs = random.choice(training_sets)
    nn.train(training_inputs, training_outputs)
    print("***",i, round(nn.calculate_total_error(training_sets), 9))
nn.test(test_set1[0][0],test_set1[0][1])
nn.test(test_set1[1][0],test_set1[1][1])
# gray_img = color2gray(img)
# h = hog(gray_img, cell_size=(8, 8), cells_per_block=(2, 2), visualise=False, nbins=9, signed_orientation=False, normalise=True)
# im2 = visualise_histogram(h, 8, 8, False)


# plt.show()
# get magnitude 

# print(get_orientation(Gx,Gy))

# hog = cv.HOGDescriptor()
# h = hog.compute(gray_img)
# print(h)

# cv.namedWindow("comparsion",1)
# cv.imshow("original", img)
# cv.imshow("gray", gray_img)
# cv.imshow("im2",im2)
# cv.imshow("myhog",np.array(training_sets[0][0]))
# cv.imshow("Gx", normalize(Gx))
# cv.imshow("Gy", normalize(Gy))
# cv.imshow("norm_mag", norm_mag_img)

# key = cv.waitKey(0)
# if key == 27:
#     cv.destroyAllWindows()
#     cv.waitKey(1)
#     cv.waitKey(1)
#     cv.waitKey(1)
#     cv.waitKey(1)
