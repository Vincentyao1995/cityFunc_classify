# coding=UTF-8
# %%
# DATA:
# 1. cifar10(binary version):https://www.cs.toronto.edu/~kriz/cifar.html
# 2. pratrained weights (vgg16.npy):https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM
 
# TO Train and test:
# 0. get data ready, get paths ready !!!
# 1. run training_and_val.py and call train() in the console
# 2. call evaluate() in the console to test

# %%


import os
import os.path

import numpy as np
import tensorflow as tf

from input_data import get_files, get_batch, get_batch_datasetVersion
import VGG
import tools
import input_data
import config
import utils
import time
# %%




dataset = config.dataset_config['thu_haidian']

IMG_W = dataset['img_w']
IMG_H = dataset['img_h']
N_CLASSES = dataset['n_classes']
BATCH_SIZE = 64
learning_rate = 0.001
MAX_STEP = 15000  # it took me about one hour to complete the training. Step is iteration
IS_PRETRAIN = True
#CAPACITY = 256
RESTORE_MODEL = False


# %%   Training
def train():
    pre_trained_weights = r'/home/vincent/Desktop/jsl_thesis/grad thesis/data/vgg16_pretrained/vgg16.npy'
    data_train_dir = r'/home/vincent/Desktop/research/vin/spp_vgg/data/UCM/train'
    data_test_dir = r'/home/vincent/Desktop/research/vin/spp_vgg/data/UCM/validation'
    train_log_dir = r'/home/vincent/Desktop/research/vin/spp_vgg/data/UCM/logs/train'
    val_log_dir = r'/home/vincent/Desktop/research/vin/spp_vgg/data/UCM/logs/val'

    if config.drop_out['switch'] == True:
        print("dropout at fc layer %f" % config.drop_out['rate'])

    with tf.name_scope('input'):
        # tra_image_batch, tra_label_batch = input_data.read_cifar10(data_dir=data_dir,
        #                                                            is_train=True,
        #                                                            batch_size=BATCH_SIZE,
        #                                                            shuffle=True)
        # val_image_batch, val_label_batch = input_data.read_cifar10(data_dir=data_dir,
        #                                                            is_train=False,
        #                                                            batch_size=BATCH_SIZE,
        #                                                            shuffle=False)
        image_train_list, label_train_list = get_files(data_train_dir)
        image_val_list, label_val_list = get_files(data_test_dir)
        # image_batch, label_batch = get_batch(image_train_list, label_train_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
        # val_image_batch, val_label_batch = get_batch(image_val_list, label_val_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
        image_batch = get_batch_datasetVersion(image_train_list, label_train_list, IMG_W, IMG_H, BATCH_SIZE)
        val_batch = get_batch_datasetVersion(image_val_list, label_val_list, IMG_W, IMG_H, BATCH_SIZE)

    # x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    # y_ = tf.placeholder(tf.int16, shape=[BATCH_SIZE, N_CLASSES])

    x = tf.placeholder(tf.float32, shape=[None, IMG_W, IMG_H, 3])
    y_ = tf.placeholder(tf.int16, shape=[None, N_CLASSES])


    # input_value = [[
    #     [[1], [2], [4], [4]],
    #     [[3], [4], [1], [2]],
    #     [[6], [2], [1], [7]],
    #     [[1], [3], [2], [8]]
    # ]]
    # input_value = np.asarray(input_value, dtype='float32')
    # input_const = tf.constant(input_value, tf.float32)
    # y = tools.spatial_pyramid_pool(input_const, 1, [int(input_const.get_shape()[1]), int(input_const.get_shape()[2])], [3])
    # with tf.Session('') as sess:
    #     y_output = sess.run(y)
    #     print(y_output)


    logits = VGG.VGG16N_SPP(x, N_CLASSES, IS_PRETRAIN)
    loss = tools.loss(logits, y_)
    accuracy = tools.accuracy(logits, y_)
    my_global_step = tf.Variable(0, name='global_step', trainable = False)
    train_op = tools.optimize(loss, learning_rate, my_global_step)

    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    sess = tf.Session()

    sess.run(init)

    # load the parameter file, assign the parameters, skip the specific layers
    tools.load_with_skip(pre_trained_weights, sess, ['fc6', 'fc7', 'fc8'])

    coord = tf.train.Coordinator()
    #threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)

    #restore older checkpoints
    if RESTORE_MODEL == True:

        print("Reading checkpoints.../n")

        log_dir = r'/home/vincent/Desktop/research/vin/spp_vgg/data/UCM/logs/train'
        model_name = r'model.ckpt-6000.meta'
        data_name = r'model.ckpt-6000'
        #restore Graph
        saver = tf.train.import_meta_graph(log_dir +os.sep + model_name)
        #restore paras
        saver.restore(sess, log_dir + os.sep + data_name)
        print("Loading checkpoints successfully!! /n")

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            tra_images, tra_labels = sess.run(image_batch)
            #tra_images, tra_labels = sess.run([image_batch, label_batch])
            #
            _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy],
                                            feed_dict={x: tra_images, y_: tra_labels})
            if step % 50 == 0 or (step + 1) == MAX_STEP:
                print('Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, tra_loss, tra_acc))
                summary_str = sess.run(summary_op, feed_dict={x: tra_images, y_: tra_labels})
                tra_summary_writer.add_summary(summary_str, step)

            if step % 200 == 0 or (step + 1) == MAX_STEP:
                val_images, val_labels = sess.run(val_batch)
                #val_images, val_labels = sess.run([val_image_batch, val_label_batch])
                val_loss, val_acc = sess.run([loss, accuracy],
                                             feed_dict={x: val_images, y_: val_labels})
                print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' % (step, val_loss, val_acc))

                summary_str = sess.run(summary_op, feed_dict={x: val_images, y_: val_labels})
                val_summary_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    #coord.join(threads)
    sess.close()


# %%   Test the accuracy on test dataset. got about 85.69% accuracy.
import math


def test_vgg_dataset():
    global dataset

    img_path = tf.placeholder(tf.string)
    img_content = tf.read_file(img_path)
    img = tf.image.decode_image(img_content, channels = 3)

    img2 = tf.image.resize_nearest_neighbor([img], [dataset['img_w'], dataset['img_h']])

    x = tf.placeholder(tf.float32, shape = [1, dataset['img_w'], dataset['img_h'],3])
    y_ = tf.placeholder(tf.int16, shape = [1, dataset['n_classes']])

    logits = VGG.VGG16N_SPP(x, dataset['n_classes'], is_pretrain = False)
    predict = tf.argmax(logits, 1)

    matrix_confusion = np.zeros((dataset['n_classes'], dataset['n_classes']))


    with tf.Session() as sess:
        # restore Graph
        #saver = tf.train.import_meta_graph(dataset['checkpoint_path']+'.meta')
        # restore paras
        saver = tf.train.Saver()
        saver.restore(sess, dataset['checkpoint_path'])
        print("Loading checkpoints successfully!! ")


        val_data_path = os.path.join(dataset['data_path'], 'validation')
        for val_class_name in os.listdir(val_data_path):
            class_path = os.path.join(val_data_path, val_class_name)
            class_index = dataset['class2label'][val_class_name]
            for val_img_name in os.listdir(class_path):
                val_img_path = os.path.join(class_path,val_img_name)
                img_content = sess.run(img2, feed_dict = {img_path: val_img_path})
                pre = sess.run(predict, feed_dict = {x:img_content})
                print(class_index, pre)
                matrix_confusion[class_index][pre] += 1
        utils.plot_confusion_matrix(matrix_confusion,
                                    normalize=True,
                                    target_names=dataset['class'],
                                    title="Confusion Matrix",
                                    saveName = dataset['out_fig_name'])
        np.savetxt(str(time.time())+'ucm_vgg_confusion_matrix', matrix_confusion)





if __name__ == '__main__':

    #calc running time

    time_start = time.time()

    train()
    #test_vgg_dataset()
    time_end = time.time()
    elapsed = time_end - time_start
    print('\n\ntime taken:',elapsed,'seconds.\n')




