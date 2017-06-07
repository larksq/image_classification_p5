"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import problem_unittests as tests
import tarfile

cifar10_dataset_folder_path = 'cifar-10-batches-py'

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

if not isfile('cifar-10-python.tar.gz'):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
        urlretrieve(
            'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
            'cifar-10-python.tar.gz',
            pbar.hook)

if not isdir(cifar10_dataset_folder_path):
    with tarfile.open('cifar-10-python.tar.gz') as tar:
        tar.extractall()
        tar.close()


tests.test_folder_path(cifar10_dataset_folder_path)

#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import helper
import numpy as np

# Explore the dataset
batch_id = 5
sample_id = 5
#helper.display_stats(cifar10_dataset_folder_path, batch_id, sample_id)

def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    # TODO: Implement Function
    max = np.ones((32, 32, 3), dtype=np.float64)*255
    X = x / max
    return X


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
#tests.test_normalize(normalize)

from sklearn import preprocessing
encoding_map = range(10)
def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    # TODO: Implement Function
    lb = preprocessing.LabelBinarizer()
    lb.fit(encoding_map)
    X = lb.transform(x)
    return X


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
#tests.test_one_hot_encode(one_hot_encode)

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)

import pickle
import problem_unittests as tests
import helper

# Load the Preprocessed Validation data
valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))




import tensorflow as tf

def neural_net_image_input(image_shape):
    """
    Return a Tensor for a batch of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    # TODO: Implement Function
    x0 = tf.placeholder(tf.float32, shape = [None, image_shape[0], image_shape[1], image_shape[2]], name = "x")
    return x0


def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    # TODO: Implement Function
    y0 = tf.placeholder(tf.float32, shape = [None, n_classes], name = "y")
    return y0


def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    # TODO: Implement Function
    keep_prob0 = tf.placeholder(tf.float32, name = "keep_prob")
    return keep_prob0


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tf.reset_default_graph()
#tests.test_nn_image_inputs(neural_net_image_input)
#tests.test_nn_label_inputs(neural_net_label_input)
#tests.test_nn_keep_prob_inputs(neural_net_keep_prob_input)

def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    # TODO: Implement Function

    # Weight and bias
    weight = tf.Variable(tf.truncated_normal ([conv_ksize[0], conv_ksize[1], int(x_tensor.shape[3]), conv_num_outputs]))
    bias = tf.Variable(tf.zeros(conv_num_outputs))
    #print('x_tensor shape:', x_tensor.shape)
    #print('conv_num_outputs:', conv_num_outputs,'conv_ksize',conv_ksize,'conv_strides', conv_strides, 'pool_ksize', pool_ksize, 'pool_strides', pool_strides)
    #Apply Convolution
    conv_layer = tf.nn.conv2d(x_tensor, weight, strides = [1, conv_strides[0], conv_strides[1], 1], padding='SAME')
    #Add bias
    conv_layer = tf.nn.bias_add(conv_layer, bias)
    #Apply activation function
    conv_layer = tf.nn.relu(conv_layer)
    #Apply max pooling
    conv_layer = tf.nn.max_pool(conv_layer, ksize = [1, pool_ksize[0], pool_ksize[1], 1], strides = [1, pool_strides[0], pool_strides[1] ,1], padding='SAME')
    return conv_layer


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
#tests.test_con_pool(conv2d_maxpool)

def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    # TODO: Implement Function
    flatten_layer = tf.contrib.layers.flatten(x_tensor)
    #print(flatten_layer)
    return flatten_layer


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
#tests.test_flatten(flatten)

def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # TODO: Implement Function
    tensor_out = tf.contrib.layers.fully_connected(x_tensor, num_outputs = num_outputs)
    #print("num_outputs:", num_outputs)
    return tensor_out


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
#tests.test_fully_conn(fully_conn)

def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # TODO: Implement Function
    tensor_out = tf.contrib.layers.fully_connected(x_tensor, num_outputs = num_outputs)
    #print("output num_outputs:", num_outputs)
    return tensor_out


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
#tests.test_output(output)

def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    # TODO: Apply 1, 2, or 3 Convolution and Max Pool layers
    #    Play around with different number of outputs, kernel size and stride
    # Function Definition from Above:
    #    conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)
    img = normalize(x)
    # print("image:", img.shape)
    conv = conv2d_maxpool(img, 4, [3,3], [1,1], [2,2], [2,2])
    conv = tf.nn.dropout(conv, keep_prob)
    # conv = conv2d_maxpool(conv, 7, [3,3], [1,1], [2,2], [2,2])
    # conv = tf.nn.dropout(conv, keep_prob)
    # conv = conv2d_maxpool(conv, 10, [3,3], [1,1], [2,2], [2,2])
    # output: 4*4*20

    # TODO: Apply a Flatten Layer
    # Function Definition from Above:
    #   flatten(x_tensor)
    conv = flatten(conv)
    print(conv.shape)

    # TODO: Apply 1, 2, or 3 Fully Connected Layers
    #    Play around with different number of outputs
    # Function Definition from Above:
    #   fully_conn(x_tensor, num_outputs)
    # conv = fully_conn(conv, int(int(conv.shape[1])/3.5))
    # print(conv.shape)
    # conv = fully_conn(conv, int(int(conv.shape[1])/3.0))
    # print(conv.shape)
    conv = fully_conn(conv, int(int(conv.shape[1])/20))
    print(conv.shape)

    # TODO: Apply an Output Layer
    #    Set this to the number of classes
    # Function Definition from Above:
    #   output(x_tensor, num_outputs)
    conv = output(conv, 10)

    # TODO: return output
    return conv


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""

##############################
## Build the Neural Network ##
##############################

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = neural_net_image_input((32, 32, 3))
y = neural_net_label_input(10)
keep_prob = neural_net_keep_prob_input()

# Model
logits = conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

#tests.test_conv_net(conv_net)


def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    # TODO: Implement Function
    # print('label_batch:', label_batch.shape)
    x = neural_net_image_input(feature_batch.shape[1:])
    y = neural_net_label_input(label_batch.shape[1])
    keep_prob = neural_net_keep_prob_input()

    logits = conv_net(x, keep_prob)
    # Name logits Tensor, so it can be loaded after training
    logits = tf.identity(logits, name='logits')

    #loss and Optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer0 = tf.train.AdamOptimizer().minimize(cost)#GradientDescentOptimizer(learning_rate=0.15).minimize(cost)
    #tf.train.AdamOptimizer().minimize(cost)
    init = tf.global_variables_initializer()

    #with tf.Session() as session:
    session.run(init)
    print('cost nn:', session.run(cost, feed_dict = {x: feature_batch, y: label_batch, keep_prob: keep_probability}))
    session.run(optimizer0, feed_dict = {x: feature_batch, y: label_batch, keep_prob: keep_probability})
    print("feature sizes:", batch_features.shape, batch_labels.shape, "valid sizes:", valid_features.shape, valid_labels.shape)
    print('label:', batch_labels[0])
    #print('After: ', session.run(cost, feed_dict = {x: feature_batch, y: label_batch, keep_prob: keep_probability}))
    print('trainning nn')

    pass


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
#tests.test_train_nn(train_neural_network)


def print_stats(session, feature_batch, label_batch, cost, accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    # TODO: Implement Function
    print("printing state")
    loss = session.run(cost, feed_dict={x: feature_batch, y:label_batch,  keep_prob: keep_probability})
    acc = session.run(accuracy, feed_dict={x: valid_features, y:valid_labels, keep_prob: keep_probability})  #???
    print("cost/loss:  ", loss)
    print("accuracy:   ", acc)

    pass


# TODO: Tune Parameters
epochs = 50
batch_size = 1
keep_probability = 0.95

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
print('Checking the Training on a Single Batch...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    #print("feature sizes:" ,batch_features.shape, batch_labels.shape, "valid sizes:", valid_features.shape, valid_labels.shape)

    # Training cycle
    for epoch in range(epochs):
        batch_i = 1
        nncount = 0
        for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
            train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
            nncount += 1
            print('nncount:',nncount)
        print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
        print_stats(sess, batch_features, batch_labels, cost, accuracy)
