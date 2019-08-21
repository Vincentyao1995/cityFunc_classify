import tensorflow as tf
import numpy as np
import math
from feature_extractor_vgg16SPP import config

drop_out = config.drop_out

# %%
def conv(layer_name, x, out_channels, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=True):
    '''Convolution op wrapper, use RELU activation after convolution
    Args:
        layer_name: e.g. conv1, pool1...
        x: input tensor, [batch_size, height, width, channels]
        out_channels: number of output channels (or comvolutional kernels)
        kernel_size: the size of convolutional kernel, VGG paper used: [3,3]
        stride: A list of ints. 1-D of length 4. VGG paper used: [1, 1, 1, 1]
        is_pretrain: if load pretrained parameters, freeze all conv layers. 
        Depending on different situations, you can just set part of conv layers to be freezed.
        the parameters of freezed layers will not change when training.
    Returns:
        4D tensor
    '''

    in_channels = x.get_shape()[-1]
    with tf.variable_scope(layer_name, reuse = tf.AUTO_REUSE):
        w = tf.get_variable(name='weights',
                            trainable=is_pretrain,
                            shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                            initializer=tf.contrib.layers.xavier_initializer())  # default is uniform distribution initialization
        b = tf.get_variable(name='biases',
                            trainable=is_pretrain,
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0))
        x = tf.nn.conv2d(x, w, stride, padding='SAME', name='conv')
        x = tf.nn.bias_add(x, b, name='bias_add')
        x = tf.nn.relu(x, name='relu')
        return x


# %%
def pool(layer_name, x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True):
    '''Pooling op
    Args:
        x: input tensor
        kernel: pooling kernel, VGG paper used [1,2,2,1], the size of kernel is 2X2
        stride: stride size, VGG paper used [1,2,2,1]
        padding:
        is_max_pool: boolen
                    if True: use max pooling
                    else: use avg pooling
    '''
    if is_max_pool:
        x = tf.nn.max_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
    else:
        x = tf.nn.avg_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
    return x
#out_pool_size = [8,6,4]
#maxpool5 = spatial_pyramid_pool(conv5,int(conv5.get_shape()[0]),[int(conv5.get_shape()[1]),int(conv5.get_shape()[2])],out_pool_size)

def spatial_pyramid_pool_old(layer_name,previous_conv, num_sample, previous_conv_size, out_pool_size):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer

    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''

    for i in range(len(out_pool_size)):

        h_strd = previous_conv_size[0] / out_pool_size[i]
        w_strd = previous_conv_size[1] / out_pool_size[i]
        h_wid = previous_conv_size[0] - h_strd * out_pool_size[i] + 1
        w_wid = previous_conv_size[1] - w_strd * out_pool_size[i] + 1
        max_pool = tf.nn.max_pool(previous_conv,
                                  ksize=[1, h_wid, w_wid, 1],
                                  strides=[1, h_strd, w_strd, 1],
                                  padding='VALID')
        max_pool = tf.nn.max_pool(previous_conv,
                                  strides=[1, h_wid, w_wid, 1],
                                  ksize=[1, 2.5, 2.5, 1],
                                  padding='VALID')
        if (i == 0):
            spp = tf.reshape(max_pool, [num_sample, -1])
        else:
            spp = tf.concat(axis=1, values=[spp, tf.reshape(max_pool, [num_sample, -1])])

    return spp

def spatial_pyramid_pool(layer_name, previous_conv, out_pool_size):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer

    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''
    '''
    代码参考：https://github.com/peace195/sppnet
    但是源码用的是下面的第二种情况，第一种情况的out大小正确，但是结果不对。
    对于n*n的input，按照stride = [window,window], kernel = [window,window]的方式，正确的只限于output <= math.ceil(n/2)，
    比如7到1234都可以，56就不行;9到12345都可以，其中4也不行。
    所以，对于生成ouput 形状不对的情况，用stride = [1,1], kernel = [in-out+1,in-out+1]的max pool来实现。
    '''

    #attention: previous 4*4 - 3*3这样的是不行的，会输出2*2；但是ucm和nwpu的dataset不存在这种情况，因为输出feature map是8*8的；（ori256/32 = 8）
    #assert: if input/out 为小数，那么余数of input/[math.ceil(input/out)]>0
    if previous_conv.shape[1].value == None:
        return previous_conv
    previous_conv_size = [int(previous_conv.shape[1]), int(previous_conv.shape[2])]
    for i in range(len(out_pool_size)):
        # print(previous_conv_size)
        h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
        # h_pad = (h_wid * out_pool_size[i] - previous_conv_size[0] + 1) / 2
        # w_pad = (w_wid * out_pool_size[i] - previous_conv_size[1] + 1) / 2

        max_pool = tf.nn.max_pool(previous_conv,
                                  ksize=[1, h_wid, w_wid, 1],
                                  strides=[1, h_wid, w_wid, 1],
                                  padding='SAME', name = layer_name)
        #assert(max_pool.shape()[1]*max_pool.shape()[2] == out_pool_size*out_pool_size)
        if int(max_pool.shape[1])*int(max_pool.shape[2]) != out_pool_size[i]*out_pool_size[i]:
            h_wid = previous_conv_size[0] - out_pool_size[i] + 1
            w_wid = previous_conv_size[1] - out_pool_size[i] + 1
            max_pool = tf.nn.max_pool(previous_conv,
                                  ksize=[1, h_wid, w_wid, 1],
                                  strides=[1, 1, 1, 1],
                                  padding='VALID', name = layer_name)
        size = int(max_pool.shape[1]) * int(max_pool.shape[2]) * int(max_pool.shape[3])
        if (i == 0):
            spp = tf.reshape(max_pool, [-1, size])
        else:
            spp = tf.concat(axis=1, values=[spp, tf.reshape(max_pool, [-1, size])])

    return spp

# %%
def batch_norm(x):
    '''Batch normlization(I didn't include the offset and scale)
    '''
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(x, [0])
    x = tf.nn.batch_normalization(x,
                                  mean=batch_mean,
                                  variance=batch_var,
                                  offset=None,
                                  scale=None,
                                  variance_epsilon=epsilon)
    return x


# %%
def FC_layer(layer_name, x, out_nodes):
    '''Wrapper for fully connected layers with RELU activation as default
    Args:
        layer_name: e.g. 'FC1', 'FC2'
        x: input feature map
        out_nodes: number of neurons for current FC layer
    '''
    shape = x.get_shape()
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value

    with tf.variable_scope(layer_name, reuse = tf.AUTO_REUSE):
        w = tf.get_variable('weights',
                            shape=[size, out_nodes],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('biases',
                            shape=[out_nodes],
                            initializer=tf.constant_initializer(0.0))
        flat_x = tf.reshape(x, [-1, size])  # flatten into 1D

        x = tf.nn.bias_add(tf.matmul(flat_x, w), b)

        global drop_out

        if drop_out['switch'] == True:
            x = tf.nn.dropout(x, drop_out['rate'])

        x = tf.nn.relu(x)
        return x


# %%
def loss(logits, labels):
    '''Compute loss
    Args:
        logits: logits tensor, [batch_size, n_classes]
        labels: one-hot labels
    '''
    with tf.name_scope('loss') as scope:
        #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross-entropy')
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross-entropy')
        loss = tf.reduce_mean(cross_entropy, name='loss') # mean loss/ final loss = mean in batch.
        tf.summary.scalar(scope + '/loss', loss)
        return loss

def loss_double(logits, labels, logits2, labels2):
    '''Compute loss
    Args:
        logits: logits tensor, [batch_size, n_classes]
        labels: one-hot labels
    '''
    with tf.name_scope('loss') as scope:
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross-entropy')
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross-entropy')

        cross_entropy2 = tf.nn.softmax_cross_entropy_with_logits(logits=logits2, labels=labels2, name='cross-entropy')

        cross_total = cross_entropy + cross_entropy2
        loss = tf.reduce_mean(cross_total, name='loss')
        tf.summary.scalar(scope + '/loss', loss)
        return loss

# %%
def accuracy(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, 
    """
    with tf.name_scope('accuracy') as scope:
        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct) * 100.0
        # tf.summary.scalar('accuracy', accuracy_)
        tf.summary.scalar(scope + '/accuracy', accuracy)
    return accuracy

def accuracy_double(logits, labels,logits2, labels2):
    """Evaluate the quality of the logits at predicting the label.
    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, 
    """
    with tf.name_scope('accuracy') as scope:
        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        correct = tf.cast(correct, tf.float32)

        correct2 = tf.equal(tf.argmax(logits2, 1), tf.argmax(labels2, 1))
        correct2 = tf.cast(correct, tf.float32)

        correct = tf.concat([correct,correct2], 0)

        accuracy = tf.reduce_mean(correct) * 100.0
        # tf.summary.scalar('accuracy', accuracy_)
        tf.summary.scalar(scope + '/accuracy', accuracy)
    return accuracy

# %%
def num_correct_prediction(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Return:
        the number of correct predictions
    """
    correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
    correct = tf.cast(correct, tf.int32)
    n_correct = tf.reduce_sum(correct)
    return n_correct

def get_wrong_num(logits, labels):
    correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
    correct = tf.cast(correct, tf.int32)
    return correct, tf.arg_max(logits, 1)

# %%
def optimize(loss, learning_rate, global_step):
    '''optimization, use Gradient Descent as default
    '''
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op


# %%
def load(data_path, session):
    data_dict = np.load(data_path, encoding='latin1').item()

    keys = sorted(data_dict.keys())
    for key in keys:
        with tf.variable_scope(key, reuse=True):
            for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                session.run(tf.get_variable(subkey).assign(data))


# %%
def test_load():
    data_path = r'/home/vincent/Desktop/jsl thesis/grad thesis/data/vgg16_pretrained/vgg16.npy'

    data_dict = np.load(data_path, encoding='latin1').item()
    keys = sorted(data_dict.keys())
    for key in keys:
        weights = data_dict[key][0]
        biases = data_dict[key][1]
        print('\n')
        print(key)
        print('weights shape: ', weights.shape)
        print('biases shape: ', biases.shape)


# %%
def load_with_skip(data_path, session, skip_layer):
    data_dict = np.load(data_path, encoding='latin1').item()
    for key in data_dict:
        if key not in skip_layer:
            with tf.variable_scope(key, reuse=True):
                for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                    session.run(tf.get_variable(subkey).assign(data))


# %%
def print_all_variables(train_only=True):
    """Print all trainable and non-trainable variables
    without tl.layers.initialize_global_variables(sess)
    Parameters
    ----------
    train_only : boolean
        If True, only print the trainable variables, otherwise, print all variables.
    """
    # tvar = tf.trainable_variables() if train_only else tf.all_variables()
    if train_only:
        t_vars = tf.trainable_variables()
        print("  [*] printing trainable variables")
    else:
        try:  # TF1.0
            t_vars = tf.global_variables()
        except:  # TF0.12
            t_vars = tf.all_variables()
        print("  [*] printing global variables")
    for idx, v in enumerate(t_vars):
        print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))

        # %%


##***** the followings are just for test the tensor size at diferent layers *********##

# %%
def weight(kernel_shape, is_uniform=True):
    ''' weight initializer
    Args:
        shape: the shape of weight
        is_uniform: boolen type.
                if True: use uniform distribution initializer
                if False: use normal distribution initizalizer
    Returns:
        weight tensor
    '''
    w = tf.get_variable(name='weights',
                        shape=kernel_shape,
                        initializer=tf.contrib.layers.xavier_initializer())
    return w


# %%
def bias(bias_shape):
    '''bias initializer
    '''
    b = tf.get_variable(name='biases',
                        shape=bias_shape,
                        initializer=tf.constant_initializer(0.0))
    return b

def ckpt2npy(FILE_PATH, OUTPUT_FILE):

    # Edit just these
    from tensorflow.python import pywrap_tensorflow

    reader = pywrap_tensorflow.NewCheckpointReader(FILE_PATH)
    var_to_shape_map = reader.get_variable_to_shape_map()
    params = {}

    for key in var_to_shape_map:
        print("tensor_name", key)
        params[key] = reader.get_tensor(key)

    np.save(OUTPUT_FILE, params)
    print("CKPT2NPY finished!")


    # %%
if __name__ == '__main__':
    FILE_PATH = r'/home/vincent/Desktop/jsl thesis/GradTest_vinny/UCM/dataset_rotated/logs/train/model.ckpt-14999'
    OUTPUT_FILE = r'/home/vincent/Desktop/jsl thesis/GradTest_vinny/UCM/dataset_rotated/logs/train/model_ckpt_15000.npy'
    ckpt2npy(FILE_PATH,OUTPUT_FILE)