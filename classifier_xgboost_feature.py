import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import matplotlib.pyplot as plt

parcel_class2label = {
    'RES': 6,
    'EDU': 1,
    'TRA': 2,
    'GRE': 3,
    'COM': 4,
    'OTH': 5,
}

def norm_feature(feas):
    n = feas.shape[0]

    for i in range(n):
        feas[i] = (feas[i]-min(feas[i]))/(max(feas[i])-min(feas[i]))
    feas = np.nan_to_num(feas)
    return feas

def plot_confusion_matrix(cm,
                          acc,
                          target_names,
                          title='Confusion matrix',
                          cmap=plt.get_cmap('jet'),
                          normalize=True,
                          saveName = 'cf.png'):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(16, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            if cm[i, j] > 0.01:
                plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        else:
            if cm[i, j] > 0.0:
                plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    saveName = 'cf_acc:' + str(round(acc,5)) + '.png'
    plt.savefig(saveName, dpi =300)
    plt.show()

def get_labels(label_path):
    label_res = []
    label_type_num = {
    'RES': 0,
    'EDU': 0,
    'TRA': 0,
    'GRE': 0,
    'COM': 0,
    'OTH': 0
    }

    with open(label_path) as f:
        line = f.readline()
        line = f.readline().strip()
        num = 0
        while line:
            num += 1
            clip_type = line.split(',')[-1]
            label_type_num[clip_type] += 1
            clip_type_label = parcel_class2label[clip_type]

            label_res.append(float(clip_type_label))
            line = f.readline().strip()
        label_res = np.array(label_res)

    return label_res, label_type_num

def concate_data(feas):
    parcel_num = feas[0].shape[0]
    new_feature_length = 0
    for fea in feas:
        new_feature_length += fea.shape[1]

    fea_res = np.zeros((parcel_num, new_feature_length))

    #input three kinds feature
    if len(feas) == 3:
        fea1 = feas[0]
        fea2 = feas[1]
        fea3 = feas[2]
        for parcel_index in range(parcel_num):
            fea_res[parcel_index][0:fea1.shape[1]] = fea1[parcel_index][:]
            fea_res[parcel_index][fea1.shape[1]:fea1.shape[1] + fea2.shape[1]] = fea2[parcel_index][:]
            fea_res[parcel_index][fea1.shape[1] + fea2.shape[1] : ] = fea3[parcel_index][:]

    # input one kind feature
    elif len(feas) == 1:
        fea_res = feas[0]

    #input two kinds feature
    else:
        fea1 = feas[0]
        fea2 = feas[1]
        for parcel_index in range(parcel_num):
            fea_res[parcel_index][0:fea1.shape[1]] = fea1[parcel_index][:]
            fea_res[parcel_index][fea1.shape[1]:fea1.shape[1] + fea2.shape[1]] = fea2[parcel_index][:]

    return fea_res

def split_test_train(fea_res, label_res):
    train_X, test_X, train_y, test_y = train_test_split(fea_res, label_res, test_size=0.05)
    return train_X, train_y, test_X, test_y


def xgboost_classifier(fea_res, label_res, label_type_num):

    #ignore features : [0,0,0....,0]
    fea_index_to_delete = []
    for fea_index in range(fea_res.shape[0]):
        cur_fea = fea_res[fea_index][0:13]
        if (cur_fea == np.zeros((1, len(cur_fea)))).all():
            fea_index_to_delete.append(fea_index)

    fea_res = np.delete(fea_res, fea_index_to_delete, axis=0)
    label_res = np.delete(label_res, fea_index_to_delete, axis=0)



    fea_train, label_train, fea_test, label_test = split_test_train(fea_res, label_res)

    model = XGBClassifier(max_depth=6, learning_rate= 0.1, objective= 'multi:softmax' )

    '''
    XGBClassifier(
        silent=0 ,#设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
        #nthread=4,# cpu 线程数 默认最大
        learning_rate= 0.3, # 如同学习率
        min_child_weight=1, 
        # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
        #，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
        #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
        max_depth=6, # 构建树的深度，越大越容易过拟合
        gamma=0,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
        subsample=1, # 随机采样训练样本 训练实例的子采样比
        max_delta_step=0,#最大增量步长，我们允许每个树的权重估计。
        colsample_bytree=1, # 生成树时进行的列采样 
        reg_lambda=1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        #reg_alpha=0, # L1 正则项参数
        #scale_pos_weight=1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
        #objective= 'multi:softmax', #多分类的问题 指定学习任务和相应的学习目标
        #num_class=10, # 类别数，多分类与 multisoftmax 并用
        n_estimators=100, #树的个数
        seed=1000 #随机种子
        #eval_metric= 'auc'
        )
        XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=160, silent=True, objective='multi:softmax')
    '''


    model.fit(fea_train, label_train)


    y_pred = model.predict(fea_test)
    print(y_pred)

    acc = accuracy_score(label_test, y_pred)
    cf = confusion_matrix(label_test, y_pred)

    if acc > 0.65:
        from sklearn.externals import joblib
        saved_model = joblib.dump(model, 'cityFunc_singleImgFea%.2f.pkl' % (acc * 100.0))

    print("accuarcy: %.2f%%" % (acc * 100.0))
    print(cf)

    # res_path = r'./dataset/logs'
    # if not os.path.exists(res_path):
    #     os.mkdir(res_path)
    #
    # res_file_path = os.path.join(res_path, 'test_log.txt')
    return cf, acc

def main(feas, norm = True):

    label_path = r'./dataset/hd_clip_parcel_type.txt'
    label_res, label_type_num = get_labels(label_path)

    fea_res = concate_data(feas)
    if norm:
        fea_res = norm_feature(fea_res)

    cf,acc = xgboost_classifier(fea_res, label_res, label_type_num)

    if acc >= 0.80:
        plot_confusion_matrix(cf, acc, target_names=list(parcel_class2label.keys()))



if __name__ is '__main__':
    main()