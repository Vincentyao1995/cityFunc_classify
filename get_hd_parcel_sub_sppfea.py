import tensorflow as tf
import tools
from spp_layer import SPPLayer
import config
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import matplotlib.pyplot as plt
import numpy as np
import math
import os


# %%
def VGG16(x, n_classes, is_pretrain=True):
    x = tools.conv('conv1_1', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = tools.conv('conv1_2', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = tools.pool('pool1', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

    x = tools.conv('conv2_1', x, 128, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = tools.conv('conv2_2', x, 128, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = tools.pool('pool2', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

    x = tools.conv('conv3_1', x, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = tools.conv('conv3_2', x, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = tools.conv('conv3_3', x, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = tools.pool('pool3', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

    x = tools.conv('conv4_1', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = tools.conv('conv4_2', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = tools.conv('conv4_3', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = tools.pool('pool3', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

    x = tools.conv('conv5_1', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = tools.conv('conv5_2', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = tools.conv('conv5_3', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = tools.pool('pool3', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

    x = tools.FC_layer('fc6', x, out_nodes=4096)
    # x = tools.batch_norm(x)
    x = tools.FC_layer('fc7', x, out_nodes=4096)
    # x = tools.batch_norm(x)
    x = tools.FC_layer('fc8', x, out_nodes=n_classes)

    return x


# %% TO get better tensorboard figures!

def VGG16N(x, n_classes, is_pretrain=True):
    # x = ops.convert_to_tensor(x, dtype=dtypes.float32)
    print(x.get_shape())
    with tf.name_scope('VGG16'):
        x = tools.conv('conv1_1', x, 3, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x = tools.conv('conv1_2', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        with tf.name_scope('pool1'):
            x = tools.pool('pool1', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x = tools.conv('conv2_1', x, 64, 128, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x = tools.conv('conv2_2', x, 128, 128, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        with tf.name_scope('pool2'):
            x = tools.pool('pool2', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x = tools.conv('conv3_1', x, 128, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x = tools.conv('conv3_2', x, 256, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x = tools.conv('conv3_3', x, 256, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        with tf.name_scope('pool3'):
            x = tools.pool('pool3', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x = tools.conv('conv4_1', x, 256, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x = tools.conv('conv4_2', x, 512, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x = tools.conv('conv4_3', x, 512, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        with tf.name_scope('pool4'):
            x = tools.pool('pool4', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x = tools.conv('conv5_1', x, 512, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x = tools.conv('conv5_2', x, 512, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x = tools.conv('conv5_3', x, 512, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        # with tf.name_scope('pool5'):
        #     x = tools.pool('pool5', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        # x = tools.FC_layer('fc6', x, out_nodes=4096)
        # with tf.name_scope('batch_norm1'):
        #     # x = tools.batch_norm(x)
        #     x = tools.FC_layer('fc7', x, out_nodes=4096)
        # with tf.name_scope('batch_norm2'):
        #     # x = tools.batch_norm(x)
        #     x = tools.FC_layer('fc8', x, out_nodes=n_classes)
        return x
        # bins = [3, 2, 1]
        # map_size = x.get_shape().as_list()[1]
        # print(map_size)
        # print(x.get_shape())
        # sppLayer = SPPLayer(bins, map_size)
        # sppool = sppLayer.spatial_pyramid_pooling(x)
        # return sppool



        # %%


if __name__ == "__main__":
    img_path = tf.placeholder(tf.string)
    # img_tensor = ops.convert_to_tensor(img_path, dtype=dtypes.string)
    img_content = tf.read_file(img_path)
    img = tf.image.decode_image(img_content, channels=3)
    img = tf.cast(img, dtype=tf.float32)
    # img = tf.image.resize_image_with_crop_or_pad(img, config.IMG_W, config.IMG_H)
    # img2 = tf.image.resize_nearest_neighbor([img], [config.IMG_H, config.IMG_W])
    # with tf.Session() as sess:
    #     mm2 = sess.run(img2,feed_dict={img_path:'hd_0578.jpg'})[0]
    #     print(mm2.shape)
    #     plt.imshow(mm2)
    #
    #     plt.show()
    x = tf.placeholder(tf.float32, shape=[1, None, None, 3])
    # print(img_tensor.get_shape())
    # y_ = tf.placeholder(tf.int16, shape=[1, 21])
    logits = VGG16N(x, 13, False)
    # predict = tf.argmax(logits, 1)
    # true_label = tf.argmax(label_batch, 1)
    # loss = tools.loss(logits, y_)
    # accuracy = tools.accuracy(logits, y_)
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(config.thu_checkpoint_path)
    print(ckpt)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print('step: ', global_step)
        i = 0
        with tf.Session() as sess:
            i = 0
            saver.restore(sess, ckpt.model_checkpoint_path)
            thu_folder_path = r'/media/jsl/675e5f7b-9f40-40ed-8ef0-adf4a2832461/data/hd_parcel/hd_parcel_ss_sub_150'
            thu_sppfea_path = r'/media/jsl/675e5f7b-9f40-40ed-8ef0-adf4a2832461/data/hd_parcel/hd_pacel_ss_sub_sppfea_150'
            for class_name in os.listdir(thu_folder_path):
                if not class_name == "hd_0072":
                    continue

                print("spp features compute start", class_name)
                image_class_folder = os.path.join(thu_folder_path, class_name)
                sppfea_class_folder = os.path.join(thu_sppfea_path, class_name)
                if not os.path.exists(sppfea_class_folder):
                    os.mkdir(sppfea_class_folder)
                for img_name in os.listdir(image_class_folder):
                    suffix = img_name[-3:]
                    if not suffix=="jpg":
                        continue
                    hd_parcel_img_name = os.path.join(image_class_folder, img_name)
                    print(hd_parcel_img_name)
                    # print(hd_parcel_img_name)
                    mm2 = sess.run(img, feed_dict={img_path: hd_parcel_img_name})
                    pre = sess.run(logits, feed_dict={x: [mm2]})
                    img_fea = pre[0]
                    img_shape = pre.shape
                    height = img_shape[1]
                    width = img_shape[2]
                    channels = img_shape[3]
                    factors = [1, 2, 3, 4]
                    results = []
                    for factor in factors:
                        result = np.zeros((factor, factor, channels))

                        split_heigth = math.ceil(height / factor)
                        split_width = math.ceil(width / factor)
                        for n in range(channels):
                            img_one_channel = img_fea[:, :, n]
                            # print(img_one_channel)
                            for i in range(factor):
                                for j in range(factor):
                                    low_height = i * split_heigth
                                    high_height = (i + 1) * split_heigth
                                    left_width = j * split_width
                                    right_width = (j + 1) * split_width

                                    img_part = img_one_channel[low_height:high_height, left_width:right_width]
                                    try:
                                        max_part = np.max(img_part)
                                    except Exception as e:
                                        print("something wrong:")
                                        max_part = 0
                                    result[i, j, n] = max_part

                        # print(result.shape)
                        # print(result)
                        results.append(result)
                    parcel_fea_path = os.path.join(sppfea_class_folder, img_name + '.npy')

                    results = np.array(results)
                    np.save(parcel_fea_path, results)
                    print("spp features saves down", parcel_fea_path)

                    # fea = np.load(parcel_fea_path)
                    # print("############read_data####################")
                    # for f in fea:
                    #     print(f.shape)
                    #     print(f)
