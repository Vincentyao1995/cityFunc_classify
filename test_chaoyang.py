import cv2
import numpy as np
import random
from tqdm import tqdm
import os

from feature_extractor_vgg16SPP import config, VGG
from sklearn.externals import joblib
import tensorflow as tf

class add_more_testData():
    '''
    this class API is to add more tests data from haidian_parcel_clip into chaoyang_test files.
    '''
    def __init__(self):
        return
    def main(self):
        add_range = [i for i in range(100,141)]

        path_npy_haidian = r'./dataset/ss_img_feas.npy'
        path_npy_chaoyang = r'./dataset/chaoyang_feas_result.npy'
        path_out = r'./dataset/chaoyang/tests_added_haidian'

        fea_haidian = np.load(path_npy_haidian)
        fea_chaoyang = np.load(path_npy_chaoyang)
        fea_new = np.concatenate((fea_chaoyang[:-1], fea_haidian[add_range[0]:add_range[-1]]), 0)
        np.save(os.path.join(path_out, 'fea_newTests.npy'), fea_new)


        label_root = r'./dataset/haidian_streetblock/hd_clip_parcel_type.txt'

        img_labels = open(label_root).readlines()[1:]
        label_dict = {}
        for label_txt in img_labels:
            label_index = int(label_txt.split(',')[0])
            label_dict.setdefault(label_index, label_txt.strip('\n').split(',')[-1])
        file_newLabelTxt = open(r'./dataset/chaoyang/tests_added_haidian/chaoyang_added_label.txt','a')
        for i in add_range:
            i_label= label_dict[i]
            str_to_write = 'hd_%04d.jpg-hd_%04d-{}\n'.format(i_label.lower()) % (i,i)
            print(str_to_write)
            file_newLabelTxt.write(str_to_write)
        file_newLabelTxt.close()

        return


class data_augment():
    def __init__(self, image_path):
        self.image_path = image_path
        #img = cv2.imread(image_path)
        self.size = 256
        return

    # 以下函数都是一些数据增强的函数
    def gamma_transform(self, img, gamma):
        gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(self.size)]

        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)

        return cv2.LUT(img, gamma_table)

    def random_gamma_transform(self, img, gamma_vari):
        log_gamma_vari = np.log(gamma_vari)

        alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)

        gamma = np.exp(alpha)

        return self.gamma_transform(img, gamma)

    def rotate(self, xb, yb, angle):

        rows, cols = xb.shape[0], xb.shape[1]

        # cols-1 and rows-1 are the coordinate limits.
        M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), angle, 1)
        xb = cv2.warpAffine(xb, M, (cols, rows))

        # M_rotate = cv2.getRotationMatrix2D((self.size / 2, self.size / 2), angle, 1)
        #
        # xb = cv2.warpAffine(xb, M_rotate, (self.size, self.size))
        #
        # yb = cv2.warpAffine(yb, M_rotate, (self.size, self.size))

        return xb, yb

    def blur(self, img):
        img = cv2.blur(img, (3, 3))

        return img

    def add_noise(self, img):
        for i in range(self.size):  # 添加点噪声

            temp_x = np.random.randint(0, img.shape[0])

            temp_y = np.random.randint(0, img.shape[1])

            img[temp_x][temp_y] = 255

        return img

    def data_augment(self, xb, yb):
        if np.random.random() < 0.25:
            xb, yb = self.rotate(xb, yb, 90)

        if np.random.random() < 0.25:
            xb, yb = self.rotate(xb, yb, 180)

        if np.random.random() < 0.25:
            xb, yb = self.rotate(xb, yb, 270)

        if np.random.random() < 0.25:
            xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转

            yb = cv2.flip(yb, 1)

        # if np.random.random() < 0.25:
        #     xb = self.random_gamma_transform(xb, 1.0)

        if np.random.random() < 0.25:
            xb = self.blur(xb)

        # 双边过滤
        if np.random.random() < 0.25:
            xb = cv2.bilateralFilter(xb, 9, 75, 75)

        # 高斯滤波
        if np.random.random() < 0.25:
            xb = cv2.GaussianBlur(xb, (5, 5), 1.5)

        if np.random.random() < 0.2:
            xb = self.add_noise(xb)

        return xb, yb

    def main(self):
        data_root = r'/home/vincent/Desktop/research/jsl_thesis/thuDateset/dataset/haidian_streetblock/hd_clip_jpg'
        out_root = r'/home/vincent/Desktop/research/jsl_thesis/thuDateset/result_analysis/sample_distribution_exp'
        label_root = r'/home/vincent/Desktop/research/jsl_thesis/thuDateset/dataset/haidian_streetblock/hd_clip_parcel_type.txt'

        if not os.path.exists(out_root):
            os.mkdir(out_root)
        img_labels = open(label_root).readlines()[1:]
        label_dict = {}
        for label_txt in img_labels:
            label_index = int(label_txt.split(',')[0])
            label_dict.setdefault(label_index, label_txt.strip('\n').split(',')[-1])

        for img_txt in sorted(os.listdir(data_root)):
            str_index = img_txt.split('.')[0].split('_')[-1]
            index = int(str_index)
            if label_dict[index] == 'COM':
                img_ogi = cv2.imread(os.path.join(data_root, img_txt))
                img_d, _ = self.data_augment(img_ogi, img_ogi)
                tmp_out_path = os.path.join(out_root, 'aug_' + img_txt)
                cv2.imwrite(tmp_out_path, img_d)
                print('data aug done! {}'.format(os.path.join(data_root, img_txt)))

        #image = cv2.imread(self.image_path)

        return

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

        #to_predict_features = ssFea_path
        to_predict_features = np.load(ssFea_path)

        train_x, train_y, test_x, test_y = split_test_train(to_predict_features, label_true)

        # 参数
        # 学习率，迭代次数，batch大小
        learning_rate = 0.001
        training_epochs = 500
        batch_size = 16
        display_step = 1

        # 网络参数
        n_hidden_1 = 256  # 第一层的特征数（神经元数）
        n_hidden_2 = 256  # 2nd layer number of features
        n_input = 19  # 输入训练x的维度
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

        pre_correct = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
        accuracy = tf.reduce_mean(tf.cast(pre_correct, tf.float32))

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        tf.summary.scalar("cost", cost)
        tf.summary.scalar('acc', accuracy)

        summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()



        with tf.Session() as sess:

            # 初始化变量

            saver = tf.train.Saver(tf.global_variables())

            sess.run(init)
            writer = tf.summary.FileWriter('./MLP_logs', sess.graph)  # 将训练日志写入到logs文件夹下

            # 迭代次数
            for epoch in range(training_epochs):
                avg_cost = 0.
                total_batch = int(len(to_predict_features) / batch_size)

                batch_x, batch_y = '', ''
                # Loop over all batches
                for i in range(total_batch):
                    # Run optimization op (backprop) and cost op (to get loss value)
                    batch_x, batch_y = get_batch(train_x, train_y, batch_size)
                    #_, c, prob_res,pred_res, summary_str = sess.run([optimizer, cost, prob, pred, summary_op], feed_dict={x: batch_x, y: batch_y})
                    _, c, prob_res, pred_res = sess.run([optimizer, cost, prob, pred],feed_dict={x: batch_x, y: batch_y})
                    # 计算平均误差
                    avg_cost += c / total_batch
                    # Display logs per epoch step
                    #writer.add_summary(summary_str, (epoch-1)*total_batch + i)  # 将日志数据写入文件

                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                          "{:.9f}".format(avg_cost))
                    # Test model
                    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
                    # Calculate accuracy
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                    print("Training Accuracy:", accuracy.eval({x: train_x, y: train_y}), '\n')

                    summary_str = sess.run(summary_op, feed_dict={x: batch_x, y: batch_y})
                    writer.add_summary(summary_str, epoch)  # 将日志数据写入文件

                if epoch % 50 == 0:
                    # Test model
                    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
                    # Calculate accuracy
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                    print("******Testing Accuracy:*******", accuracy.eval({x: test_x, y: test_y}))

                    # summary_str = sess.run(summary_op, feed_dict={x: batch_x, y: batch_y})

            print("Optimization Finished!")
            saver.save(sess, './MLP_log_cityFunc_model.ckpt', global_step=epoch)


    def nn_classify_ssFeas(self, ssFea_path, model_path, file_ref_path, res_path):
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
        n_input = 19  # 输入训练x的维度
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


        file = open(res_path, 'w')


        file_ref = open(file_ref_path, 'r')
        refs = file_ref.readlines()

        file.write('label--index--EDU,TRA,GRE,COM,OTH,RES\n')

        parcel_class2label = {
            6:'RES',
            1:'EDU',
            2:'TRA',
            3:'GRE',
            4:'COM',
            5:'OTH',
        }

        with tf.Session() as sess:
            # restore Graph
            # restore paras
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            print('loading model successfully! %s' % model_path)

            pred_logits = sess.run(pred, feed_dict={x: to_predict_features})
            prob_res = sess.run(prob, feed_dict={x: to_predict_features})

            for i, prob in enumerate(prob_res[:-1]):
                prob = list(prob)
                prob_tmp = list(prob.copy())
                max_first = max(prob_tmp)
                prob_tmp.remove(max_first)
                max_second = max(prob_tmp)
                prob_tmp.remove(max_second)
                max_third = max(prob_tmp)
                top_list = []
                top_list.append((parcel_class2label[prob.index(max_first)+1], max_first))
                top_list.append((parcel_class2label[prob.index(max_second)+1], max_second))
                top_list.append((parcel_class2label[prob.index(max_third)+1], max_third))

                file.write(refs[i].strip('\n') + '--------' + str(top_list) + '\n')
            print('file output at %s' % res_path)
            file.close()
            file_ref.close()


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

class mutiFea_fusion():
    """
    这个类是用来fuse海淀和朝阳的test feature的，生成朝阳的cityFunc 特征，之后截取海淀的和朝阳的这个特征，生成一个新的.npy文件，成为cityFunc6特征
    首先：mutiFea_fusion.generate_chaoyang_img_txt()，用来生成朝阳区图像的路径txt。
    2）generate_cityFunc() 读取txt路径，并且用cityFunc模型输出朝阳对应的fea.
    3) fuse_haidian_chaoyang()，对输出的朝阳.npy cityFunc6，融合海淀的cityFunc6特征，并且输出.npy文件保存。
    
    """

    def __init__(self):
        return
    def fusion(self, *args, **kwargs):
        feas = ''
        for arg in args:
            if os.path.isfile(arg):
                fea_tmp = np.load(arg)
                norm_fea = self.norm_feature(fea_tmp)
                if feas == '':
                    feas = norm_fea
                    continue
                else:
                    feas = np.concatenate([feas, norm_fea],1)
        return feas

    def fuse_haidian_chaoyang(self):

        add_range = [i for i in range(100, 141)]

        path_npy_haidian = r'./dataset/parcel_imgFea_cityFuncLogits.npy'
        path_npy_chaoyang = r'./dataset/chaoyang/tests_added_haidian/chaoyangFeas_cityFuncLogits.npy'
        path_out = r'./dataset/chaoyang/tests_added_haidian/added_fea_cityFunc6.npy'

        fea_haidian = np.load(path_npy_haidian)
        fea_chaoyang = np.load(path_npy_chaoyang)
        fea_new = np.concatenate((fea_chaoyang.squeeze(), fea_haidian[add_range[0]:add_range[-1]]), 0)
        np.save(path_out, fea_new)
        print('new npy feature saved: {}'.format(path_out))

    def generate_chaoyang_img_txt(self):
        root_path = r'/home/vincent/Desktop/research/jsl_thesis/thuDateset/dataset/chaoyang/imgs/train'
        res_txt = os.path.join(root_path, 'chaoyang_imgs.txt')
        file_txt = open(res_txt, 'w')
        for file in sorted(os.listdir(root_path)):
            if file.split('.')[-1] == 'jpg':
                file_txt.write(os.path.join(root_path, file))
                file_txt.write('\n')
        file_txt.close()
        print('generated txt file: {}'.format(res_txt))

    def generate_cityFunc(self):
        '''
        输入朝阳区域的影像，使用cityFunc模型提取其fea并且保存成.npy文件；之后使用multi_fea_fusion函数，读取两个.npy文件，抽取对应的fea融合成新的.npy。
        :return: 
        '''

        dataset_choice = 'cityFunc'

        IMG_W = 256
        IMG_H = 256
        N_CLASSES = 6


        RESTORE_MODEL = True

        cityFunc_model_path = r'./dataset/cityFuncDataset/logs/train/cityFuncSpp8:2/model.ckpt-14999'
        dataset_choice = 'cityFunc'

        chaoyang_imgs_txt = r'./dataset/chaoyang/imgs/train/chaoyang_imgs.txt'
        chaoyang_imgs = open(chaoyang_imgs_txt, 'r')
        res_path = r'./dataset/chaoyang/tests_added_haidian/chaoyangFeas_cityFuncLogits.npy'

        img_path = tf.placeholder(tf.string)
        img_content = tf.read_file(img_path)
        img = tf.image.decode_image(img_content, channels=3)
        img2 = tf.image.resize_nearest_neighbor([img], [IMG_W, IMG_H])

        x = tf.placeholder(tf.float32, shape=[None, IMG_W, IMG_H, 3])
        y_ = tf.placeholder(tf.int16, shape=[None, N_CLASSES])

        logits = VGG.VGG16N_SPP(x, N_CLASSES)

        init = tf.global_variables_initializer()
        sess = tf.Session()

        sess.run(init)

        # restore older checkpoints
        if RESTORE_MODEL == True:
            print("Reading checkpoints.../n")

            model_name = cityFunc_model_path + '.meta'
            data_name = cityFunc_model_path

            # restore Graph
            saver = tf.train.import_meta_graph(model_name)
            # restore paras
            saver.restore(sess, data_name)
            print("Loading checkpoints successfully!! /n")

        feas_res_chaoyang = []
        for img in chaoyang_imgs:
            '''这里需要遍历朝阳区的img，然后用cityFunc模型输出对应的logits.npy，之后用fusion函数合成两个.npy文件。'''
            img = img.strip('\n')
            img_content = sess.run(img2, feed_dict={img_path: img})
            logits_result = sess.run(logits, feed_dict={x: img_content})

            feas_res_chaoyang.append(logits_result)
        feas_res_chaoyang = np.array(feas_res_chaoyang)

        np.save(res_path, feas_res_chaoyang)
        print('saved feas res chaoyang cityFunc6 at {}'.format(res_path))
        sess.close()

    def concate_data(feas):
        parcel_num = feas[0].shape[0]
        new_feature_length = 0
        for fea in feas:
            new_feature_length += fea.shape[1]

        fea_res = np.zeros((parcel_num, new_feature_length))

        # input three kinds feature
        if len(feas) == 3:
            fea1 = feas[0]
            fea2 = feas[1]
            fea3 = feas[2]
            for parcel_index in range(parcel_num):
                fea_res[parcel_index][0:fea1.shape[1]] = fea1[parcel_index][:]
                fea_res[parcel_index][fea1.shape[1]:fea1.shape[1] + fea2.shape[1]] = fea2[parcel_index][:]
                fea_res[parcel_index][fea1.shape[1] + fea2.shape[1]:] = fea3[parcel_index][:]

        # input one kind feature
        elif len(feas) == 1:
            fea_res = feas[0]

        # input two kinds feature
        else:
            fea1 = feas[0]
            fea2 = feas[1]
            for parcel_index in range(parcel_num):
                fea_res[parcel_index][0:fea1.shape[1]] = fea1[parcel_index][:]
                fea_res[parcel_index][fea1.shape[1]:fea1.shape[1] + fea2.shape[1]] = fea2[parcel_index][:]

        return fea_res

    def norm_feature(self, feas):
        n = feas.shape[0]

        for i in range(n):
            feas[i] = (feas[i] - min(feas[i])) / (max(feas[i]) - min(feas[i]) + 0.0000001)
        feas = np.nan_to_num(feas)
        return feas

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
    ssFea_added_path =r'./dataset/chaoyang/tests_added_haidian/fea_newTests.npy'

    nn_ss_train_feaPath = r'/home/vincent/Desktop/research/jsl_thesis/thuDateset/dataset/ss_img_feas.npy'
    nn_ss_label_feaPath = r'/home/vincent/Desktop/research/jsl_thesis/thuDateset/dataset/hd_clip_parcel_type.txt'
    nn_model_path = r'./trained_MLP_model/saved_model_res/MLP_log_model.ckpt-499'
    file_ref_path = r'./trained_MLP_model/saved_model_res/test_res_chaoyang/chaoyang_label.txt'
    added_file_ref_path = r'./dataset/chaoyang/tests_added_haidian/chaoyang_added_label.txt'
    res_path = r'./test_nn_added_chaoyang.txt'

    #to_classify_chaoyang.convert_tif2jpg(input_filepath, output_filepath)
    #to_classify_chaoyang.process_name_class(subRegion_path, subRegion_spp_path, output_filepath)

    #to_classify_chaoyang.nn_train_ssFeas(nn_ss_train_feaPath, nn_ss_label_feaPath)
    # to_classify_chaoyang.nn_classify_ssFeas(ssFea_added_path, nn_model_path, added_file_ref_path, res_path)

    #this API could do dataAugmentation operation on COM class.
    #dataAug = data_augment('').main()


    # add_data = add_more_testData()
    # add_data.main()
    """
    这个类是用来fuse海淀和朝阳的test feature的，生成朝阳的cityFunc 特征，之后截取海淀的和朝阳的这个特征，生成一个新的.npy文件，成为cityFunc6特征
    首先：mutiFea_fusion.generate_chaoyang_img_txt()，用来生成朝阳区图像的路径txt。
    2）generate_cityFunc() 读取txt路径，并且用cityFunc模型输出朝阳对应的fea.
    3) fuse_haidian_chaoyang()，对输出的朝阳.npy cityFunc6，融合海淀的cityFunc6特征，并且输出.npy文件保存。

    """
    multi_fuse = mutiFea_fusion()

    # multi_fuse.generate_chaoyang_img_txt()
    # multi_fuse.generate_cityFunc()
    # multi_fuse.fuse_haidian_chaoyang()

    addedTest_cityFunc6_fea = r'./dataset/chaoyang/tests_added_haidian/added_fea_cityFunc6.npy'
    addedTest_ss13_fea = r'./dataset/chaoyang/tests_added_haidian/fea_newTests.npy'
    fusion_test_cityFunc_fea19 = multi_fuse.fusion(addedTest_ss13_fea, addedTest_cityFunc6_fea)
    fusion_test_cityFunc_fea19_path = r'./tmp/fusion_test_cityFunc_fea19.npy'
    np.save(fusion_test_cityFunc_fea19_path, fusion_test_cityFunc_fea19)

    nn_cityFunc6_train_feaPath = r'/home/vincent/Desktop/research/jsl_thesis/thuDateset/dataset/parcel_imgFea_cityFuncLogits.npy'
    nn_cityFunc6_label_feaPath = r'/home/vincent/Desktop/research/jsl_thesis/thuDateset/dataset/hd_clip_parcel_type.txt'
    fusion_train_cityFunc_fea19 = multi_fuse.fusion(nn_ss_train_feaPath, nn_cityFunc6_train_feaPath)
    fusion_train_cityFunc_fea19_path = './tmp/fusion_train_cityFunc_fea19.npy'
    np.save(fusion_train_cityFunc_fea19_path, fusion_train_cityFunc_fea19)

    # to_classify_chaoyang.nn_train_ssFeas(fusion_train_cityFunc_fea19_path, nn_ss_label_feaPath)
    to_classify_chaoyang.nn_classify_ssFeas(fusion_test_cityFunc_fea19_path,
                                            r'./trained_MLP_model/saved_model_res/MLP_log_cityFunc_model.ckpt-499',
                                            r'./dataset/chaoyang/tests_added_haidian/chaoyang_added_label.txt',
                                            r'./dataset/chaoyang/tests_added_haidian/fea6+13_added_test.txt'
                                            )
