# cityFunc_classify

这个程序是用来使用VGG16+SPP1234结构提取出basic scenes的特征，之后使用xgboost回归此特征；

此称为cls1；

之后在street block in haidian district, Beijing，上面用selective search 算法生成sub-regions，并且使用cls1进行分类；

之后使用VBoW模型进行统计，形成街区（ss）的features。

之后再用xgboost训练ss+poi+mobike的特征；形成最终的cityFunc 分类。
