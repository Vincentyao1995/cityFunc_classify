import cv2
import numpy as np
import random
from tqdm import tqdm
import os

from feature_extractor_vgg16SPP import config, VGG
from sklearn.externals import joblib
import tensorflow as tf


class classifyChaoyang():
    def __init__(self, img_path, output_path):
        self._img_path = img_path
        self._output_path = output_path
        self._size = 256
        #self._saved_model = joblib.load('./dataset/chaoyang/cityFunc_singleFea.pkl')
        return

    def classify_single_img(self, image_path, dataset_choice='cityFunc'):
        '''
            This function is to test chaoyang single img classification. mainly to extract img features from single pic using vgg16+spp+softmax model, results write in .npy files, check '../dataset/haidian_streetblock/hd_clip_jpg_SPPfeature' 

            To change model, change: config.thu_checkpoint_path = r'../dataset/THUDataset/logs/train/model.ckpt-14999'

        '''
        dataset_choice = 'cityFunc'
        dataset = config.dataset_config[dataset_choice]

        img_path = tf.placeholder(tf.string)
        img_content = tf.read_file(img_path)
        img = tf.image.decode_image(img_content, channels=3)

        img2 = tf.image.resize_nearest_neighbor([img], [dataset['img_w'], dataset['img_h']])

        x = tf.placeholder(tf.float32, shape=[1, dataset['img_w'], dataset['img_h'], 3])
        y_ = tf.placeholder(tf.int16, shape=[1, dataset['n_classes']])

        logits = VGG.VGG16N_SPP(x, dataset['n_classes'], is_pretrain=False)
        feature = tf.nn.softmax(logits)
        class_res = tf.argmax(feature, 1)

        with tf.Session() as sess:
            # restore Graph
            # restore paras
            saver = tf.train.Saver()
            saver.restore(sess, dataset['checkpoint_path'])
            #print("Loading trained model successfully!! ")

            img_content = sess.run(img2, feed_dict={img_path: image_path})
            fea = sess.run(feature, feed_dict={x: img_content})
            res = sess.run(class_res, feed_dict={x: img_content})

        tf.reset_default_graph()  # 重置默认图
        return res


    def generate_imgs(self, image_num = 500):
        '''
        该函数用来生成用于分类的小图像，切图方法为随机切图采样，输入小图像后会使用def classify_single_img()函数来做图像分类，最后输出这些随机裁剪小图像的labels在 label image上展示出。
        :param image_num: 生成样本的个数
        :return:
        '''

        # 用来记录所有的子图的数目
        images_path = [self._img_path]
        size = self._size

        # 每张图片生成子图的个数

        for i in tqdm(range(len(images_path))):
            count = 0
            image = cv2.imread(images_path[i])
            # cv2.imwrite('pngImg.png', image)
            # return
            X_height, X_width = image.shape[0], image.shape[1]
            new_img = image.copy()
            new_img[:] = [255, 255, 255]
            while count < image_num:
                random_width = random.randint(0, X_width - size - 1)
                random_height = random.randint(0, X_height - size - 1)
                image_ogi = image[random_height: random_height + size, random_width: random_width + size, :]

                tmp_imgPath = self._output_path + '%05d.png' % count
                cv2.imwrite(tmp_imgPath, image_ogi)
                cls = self.classify_single_img(tmp_imgPath)[0]
                new_img[random_height: random_height + size, random_width: random_width + size, :] = cls

                print('img: %s, class: %d' % (tmp_imgPath, cls))

                count += 1
            label_img_path = os.path.join(self._output_path, 'labelImg.png')
            cv2.imwrite(label_img_path, new_img)
            color_img_path = os.path.join(self._output_path, 'colorImg.png')
            self.color_annotation(label_img_path, color_img_path)

    def classify_whole_img(self):
        '''
        这个函数用于规则地裁剪划分小图像，之后规则地按顺序来划分一整副图像（可以自己在函数内部通过更改clip_height_num clip_width_num参数，来确定裁剪分类图像数目。）
        '''
        whole_img_path = self._img_path
        whole_img = cv2.imread(whole_img_path)

        new_img = whole_img.copy()
        new_img[:] = [255, 255, 255]

        X_height, X_width = whole_img.shape[0], whole_img.shape[1]

        clip_height_num = X_height//self._size
        clip_width_num = X_width//self._size
        clip_height_num = 15
        clip_width_num = 15

        for i in tqdm(range(clip_height_num)):
            for j in range(clip_width_num):
                to_classify_img = whole_img[i*self._size:(i+1)*self._size, j*self._size:(j+1)*self._size]
                tmp_singleImg_path = os.path.join(self._output_path, 'tmp', 'h%d_w%d.png' % (i, j))

                if not os.path.exists(tmp_singleImg_path):
                    cv2.imwrite(tmp_singleImg_path, to_classify_img)
                cls = self.classify_single_img(tmp_singleImg_path)[0]
                new_img[i*self._size:(i+1)*self._size, j*self._size:(j+1)*self._size] = cls
                print('img: %s, class: %d' % (tmp_singleImg_path, cls))

        label_img_path = os.path.join(self._output_path, 'res_labelImg.png')
        cv2.imwrite(label_img_path, new_img)
        color_img_path = os.path.join(self._output_path, 'res_colorImg.png')
        self.color_annotation(label_img_path, color_img_path)

    def classify_whole_img_pixel(self):
        '''
        这个函数用于规则地裁剪划分小图像，并且按照步长为1地来遍历，并且建立基于像素的分类投票机制，来给每一个像素进行分类，算是patch-based CNN
        '''

        whole_img_path = self._img_path
        whole_img = cv2.imread(whole_img_path)
        set_width = set_height = 1000
        range_width = [set_width,set_width+400]
        range_height = [set_height,set_height+400]
        whole_img = whole_img[range_width[0]:range_width[1], range_height[0]:range_height[1]]

        new_img = whole_img.copy()
        new_img[:] = [255, 255, 255]

        res = []
        for i in range(whole_img.shape[0]):
            res.append([])
            for j in range(whole_img.shape[1]):
                res[i].append([0, 0, 0, 0, 0])

        X_height, X_width = whole_img.shape[0], whole_img.shape[1]

        for i in tqdm(range(0, whole_img.shape[0]-256, 2)):
            for j in range(0, whole_img.shape[1]-256, 2):
                to_classify_img = whole_img[i:i+256, j:j+256]
                tmp_singleImg_path = os.path.join(self._output_path, 'tmp', 'h%d_w%d.png' % (i, j))

                if not os.path.exists(tmp_singleImg_path):
                    cv2.imwrite(tmp_singleImg_path, to_classify_img)
                cls = self.classify_single_img(tmp_singleImg_path)[0]
                #将当前图片分类结果写入res中，这是一个二维数组。
                for ti in range(i, i+256):
                    for tj in range(j, j+256):
                        res[ti][tj][cls] += 1
                        pixel_res = res[ti][tj]
                        new_img[ti][tj] = pixel_res.index(max(pixel_res))

                print('img: %s, class: %d' % (tmp_singleImg_path, cls))
        self.save_clip_img(range_width, range_height)
        label_img_path = os.path.join(self._output_path, 'res_labelImg_pixel.png')
        cv2.imwrite(label_img_path, new_img)
        color_img_path = os.path.join(self._output_path, 'res_colorImg_pixel.png')
        self.color_annotation(label_img_path, color_img_path)

    def convert_tif2jpg(self, input_filepath, output_filepath):
        #this function is to converting all images in input filepath to .jpg img into output filepath
        for file in os.listdir(input_filepath):
            if file.split('.')[-1] in ['bmp', 'tif']:
                img_path = os.path.join(input_filepath, file)
                out_img_path = os.path.join(output_filepath, file.split('.')[0] + '.jpg')
                img = cv2.imread(img_path)
                if not os.path.exists(out_img_path):
                    cv2.imwrite(out_img_path, img)
                    print('write %s done!' % out_img_path)

        return

    def process_name_class(self, sub_region_path, sub_region_spp_path, img_path):
        file_index = 0
        res_file_path = os.path.join(img_path, 'chaoyang_label.txt')
        res_file = open(res_file_path, 'w')
        for file in sorted(os.listdir(img_path)):
            if file.split('.')[-1] != 'jpg':
                continue
            res_file.writelines(file + '-chaoyang_%.3d-%s' % (file_index, file.split('_')[0]))
            res_file.writelines('\n')
            print(file + '-chaoyang_%.3d-%s' % (file_index, file.split('_')[0]))
            file_index += 1
            src_subregion_path = os.path.join(sub_region_path, file.split('.')[0])
            src_subregion_spp_path = os.path.join(sub_region_spp_path, file.split('.')[0])
            if not os.path.exists(src_subregion_path):
                continue
            new_subregion_path = os.path.join(src_subregion_path.replace(file.split('.')[0], 'chaoyang_%.3d' % file_index))
            new_subregion_spp_path = os.path.join(src_subregion_spp_path.replace(file.split('.')[0], 'chaoyang_%.3d' % file_index))
            os.rename(src_subregion_path, new_subregion_path)
            os.rename(src_subregion_spp_path, new_subregion_spp_path)

    def xgboost_classify_ssFeas(self, xgboost_model_path, ssFea_path):
        from sklearn.externals import joblib
        import numpy as np
        def get_labels():

            parcel_class2label = {
                'RES': 6,
                'EDU': 1,
                'TRA': 2,
                'GRE': 3,
                'COM': 4,
                'OTH': 5,
            }
            label_res = []
            label_path = r'./dataset/chaoyang/imgs/train/chaoyang_label.txt'
            label_file = open(label_path, 'r')
            lines = label_file.readlines()
            for line in lines:
                clip_type = line.replace('\n','').split('-')[-1]
                clip_type = clip_type.upper()
                clip_type_label = parcel_class2label[clip_type]
                label_res.append(int(clip_type_label))
            return label_res

        label_true = get_labels()
        to_predict_features = np.load(ssFea_path)

        xgb_model = joblib.load(xgboost_model_path)
        label_pre = xgb_model.predict(to_predict_features)
        from sklearn.metrics import accuracy_score, confusion_matrix
        label_pre = label_pre[:len(label_true)]
        acc, cf = accuracy_score(label_true, label_pre), confusion_matrix(label_true, label_pre)
        print(acc, cf)

    def nn_train_ssFeas(self, ssFea_path, label_path):
        import numpy as np
        from sklearn.model_selection import train_test_split

        # def get_labels(label_path = label_path):
        #     parcel_class2label = {
        #         'RES': 6,
        #         'EDU': 1,
        #         'TRA': 2,
        #         'GRE': 3,
        #         'COM': 4,
        #         'OTH': 5,
        #     }
        #     label_res = []
        #     #label_path = r'./dataset/chaoyang/imgs/train/chaoyang_label.txt'
        #     label_file = open(label_path, 'r')
        #     lines = label_file.readlines()
        #     for line in lines:
        #         clip_type = line.replace('\n', '').split('-')[-1]
        #         clip_type = clip_type.upper()
        #         clip_type_label = parcel_class2label[clip_type]
        #         label_res.append(int(clip_type_label))
        #     return label_res
        def get_labels(label_path = label_path):
            label_res = []
            label_type_num = {
                'RES': 0,
                'EDU': 0,
                'TRA': 0,
                'GRE': 0,
                'COM': 0,
                'OTH': 0
            }
            parcel_class2label = {
                'RES': 6,
                'EDU': 1,
                'TRA': 2,
                'GRE': 3,
                'COM': 4,
                'OTH': 5,
            }
            basic_label = [0,0,0,0,0,0]

            with open(label_path) as f:
                line = f.readline()
                line = f.readline().strip()
                num = 0
                while line:
                    num += 1
                    clip_type = line.split(',')[-1]
                    label_type_num[clip_type] += 1
                    clip_type_label = parcel_class2label[clip_type]

                    tmp_label = basic_label.copy()
                    tmp_label[int(clip_type_label)-1] += 1

                    label_res.append(tmp_label)
                    line = f.readline().strip()
                label_res = np.array(label_res)

            return label_res, label_type_num

        # 创建多层感知机模型
        def multilayer_perceptron(x, weights, biases):
            # Hidden layer with RELU activation
            layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
            layer_1 = tf.nn.relu(layer_1)
            # Hidden layer with RELU activation
            layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
            layer_2 = tf.nn.relu(layer_2)
            # Output layer with linear activation
            out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
            return out_layer

        def get_batch(x, x_true, batch_size):
            rnd_indices = np.random.randint(0, len(x), batch_size)
            x_batch = x[rnd_indices]
            y_batch = x_true[rnd_indices]
            return x_batch, y_batch

        def split_test_train(fea_res, label_res):
            train_X, test_X, train_y, test_y = train_test_split(fea_res, label_res, test_size=0.10)
            return train_X, train_y, test_X, test_y

        label_true, _ = get_labels(label_path)
        to_predict_features = np.load(ssFea_path)
        train_x, train_y, test_x, test_y = split_test_train(to_predict_features, label_true)

        # 参数
        # 学习率，迭代次数，batch大小
        learning_rate = 0.001
        training_epochs = 500
        batch_size = 8
        display_step = 1

        # 网络参数
        n_hidden_1 = 256  # 第一层的特征数（神经元数）
        n_hidden_2 = 256  # 2nd layer number of features
        n_input = 13  # 输入训练x的维度
        n_classes = 6  # 输出类别数目

        # tf 图的输入
        x = tf.placeholder("float", [None, n_input])
        y = tf.placeholder("float", [None, n_classes])

        # 权重、偏置参数
        weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

        # 创建模型
        pred = multilayer_perceptron(x, weights, biases)
        prob = tf.nn.softmax(pred)

        # 定义 loss 和 optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # 初始化变量
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.global_variables())


        with tf.Session() as sess:
            sess.run(init)

            # 迭代次数
            for epoch in range(training_epochs):
                avg_cost = 0.
                total_batch = int(len(to_predict_features) / batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    batch_x, batch_y = get_batch(train_x, train_y, batch_size)
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c, prob_res,pred_res = sess.run([optimizer, cost, prob, pred], feed_dict={x: batch_x, y: batch_y})
                    # 计算平均误差
                    avg_cost += c / total_batch
                # Display logs per epoch step
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                          "{:.9f}".format(avg_cost))
                    # Test model
                    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
                    # Calculate accuracy
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                    print("Training Accuracy:", accuracy.eval({x: train_x, y: train_y}), '\n')
                if epoch % 50 == 0:
                    # Test model
                    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
                    # Calculate accuracy
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                    print("******Testing Accuracy:*******", accuracy.eval({x: test_x, y: test_y}))

            print("Optimization Finished!")
            saver.save(sess, './MLPmodel.ckpt', global_step=epoch)
            #
            # # Test model
            # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            # # Calculate accuracy
            # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            # print("******Testing Accuracy:*******", accuracy.eval({x: test_x, y: test_y}))

    def nn_classify_ssFeas(self, ssFea_path, model_path):
        # 创建多层感知机模型
        def multilayer_perceptron(x, weights, biases):
            # Hidden layer with RELU activation
            layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
            layer_1 = tf.nn.relu(layer_1)
            # Hidden layer with RELU activation
            layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
            layer_2 = tf.nn.relu(layer_2)
            # Output layer with linear activation
            out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
            return out_layer

        to_predict_features = np.load(ssFea_path)


        n_hidden_1 = 256  # 第一层的特征数（神经元数）
        n_hidden_2 = 256  # 2nd layer number of features
        n_input = 13  # 输入训练x的维度
        n_classes = 6  # 输出类别数目

        # tf 图的输入
        x = tf.placeholder("float", [None, n_input])
        y = tf.placeholder("float", [None, n_classes])

        # 权重、偏置参数
        weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

        # 创建模型
        pred = multilayer_perceptron(x, weights, biases)
        # relu_pred = tf.nn.relu(pred[:-1])
        pred = tf.multiply(pred, tf.constant(0.001,dtype=tf.float32))
        prob = tf.nn.softmax(pred, 1)


        with tf.Session() as sess:
            # restore Graph
            # restore paras
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            print('loading model successfully! %s' % model_path)

            pred_logits = sess.run(pred, feed_dict={x: to_predict_features})
            prob_res = sess.run(prob, feed_dict={x: to_predict_features})
            print(prob_res)


    def color_annotation(self, label_path, output_path):

        '''

        给class图上色

        '''

        img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        color = np.ones([img.shape[0], img.shape[1], 3])

        color[img == 0] = [255, 255, 255]  # 其他，白色，0
        color[img == 1] = [0, 255, 0]  # 植被，绿色，1
        color[img == 2] = [0, 0, 0]  # 道路，黑色，2
        color[img == 3] = [131, 139, 139]  # 建筑，黄色，3
        color[img == 4] = [139, 69, 19]  # 水体，蓝色，4

        cv2.imwrite(output_path, color)
    def save_clip_img(self, height_range, width_range):
        whole_img_path = self._img_path
        whole_img = cv2.imread(whole_img_path)
        new_img = whole_img[height_range[0]:height_range[1], width_range[0]:width_range[1]]
        new_img_path = os.path.join(self._output_path, 'ori_clip_img_h%d-%d_w%d-%d.png' %(height_range[0], height_range[1], width_range[0], width_range[1]))
        cv2.imwrite(new_img_path, new_img)
        print('saved %s' % new_img_path)

if __name__ == '__main__':

    img_path = './beijing_level18_1_1_clip.tif'
    output_path = './dataset/chaoyang/'
    to_classify_chaoyang = classifyChaoyang(img_path, output_path)

    input_filepath = r'/home/vincent/Desktop/research/jsl_thesis/thuDateset/dataset/chaoyang/imgs'
    output_filepath = r'/home/vincent/Desktop/research/jsl_thesis/thuDateset/dataset/chaoyang/imgs/train'
    subRegion_path = r'/home/vincent/Desktop/research/jsl_thesis/thuDateset/dataset/chaoyang/imgs/sub_regions'
    subRegion_spp_path = r'/home/vincent/Desktop/research/jsl_thesis/thuDateset/dataset/chaoyang/imgs/sub_regions_spp'

    xgboost_model_path = r'/home/vincent/Desktop/research/jsl_thesis/thuDateset/cityFunc_singleImgFea68.52.pkl'
    ssFea_path = r'./dataset/chaoyang_feas_result.npy'

    nn_ss_train_feaPath = r'/home/vincent/Desktop/research/jsl_thesis/thuDateset/dataset/ss_img_feas.npy'
    nn_ss_label_feaPath = r'/home/vincent/Desktop/research/jsl_thesis/thuDateset/dataset/hd_clip_parcel_type.txt'
    nn_model_path = './MLPmodel.ckpt-499'

    #to_classify_chaoyang.convert_tif2jpg(input_filepath, output_filepath)
    #to_classify_chaoyang.process_name_class(subRegion_path, subRegion_spp_path, output_filepath)

    #to_classify_chaoyang.nn_train_ssFeas(nn_ss_train_feaPath, nn_ss_label_feaPath)
    to_classify_chaoyang.nn_classify_ssFeas(ssFea_path, nn_model_path)