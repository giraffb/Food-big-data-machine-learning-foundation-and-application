# 通用库
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
import matplotlib.pyplot as plt

im_shape = (299,299)
TRAINING_DIR = '../input/amazon-fruits-small/ds_frutas_am/train'
TEST_DIR = '../input/amazon-fruits-small/ds_frutas_am/test'
seed = 10
BATCH_SIZE = 16
# 使用 keras 的 ImageGenerator 和 flow_from_directoty
data_generator = ImageDataGenerator(
        validation_split=0.2,  # 验证集比例
        rotation_range=20,  # 随机旋转角度范围
        width_shift_range=0.2,  # 随机水平位移范围
        height_shift_range=0.2,  # 随机垂直位移范围
        preprocessing_function=preprocess_input,  # 预处理函数
        shear_range=0.2,  # 剪切强度
        zoom_range=0.2,  # 随机缩放范围
        horizontal_flip=True,  # 水平翻转
        fill_mode='nearest')  # 填充模式
val_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)
# 训练部分的数据生成器
train_generator = data_generator.flow_from_directory(TRAINING_DIR,  # 训练集路径
                                          target_size=im_shape, # 图像尺寸
                                             shuffle=True,  # 是否打乱顺序
                                                   seed=seed,  # 随机种子
                                    class_mode='categorical',  # 类别模式
                                    batch_size=BATCH_SIZE,  # 批量大小
                                    subset="training")  # 数据子集：训练集
# 验证部分的数据生成器
validation_generator = val_data_generator.flow_from_directory(TRAINING_DIR,  # 训练集路径
                                      target_size=im_shape,  # 图像尺寸
                                             shuffle=False,  # 不打乱顺序
                                                 seed=seed,  # 随机种子
                                  class_mode='categorical',  # 类别模式
                                     batch_size=BATCH_SIZE,  # 批量大小
                                  subset="validation")  # 数据子集：验证集
# 测试数据的生成器
test_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_generator.flow_from_directory(TEST_DIR,  # 测试集路径
                                       target_size=im_shape,  # 图像尺寸
                                              shuffle=False,  # 不打乱顺序
                                                  seed=seed,  # 随机种子
                                   class_mode='categorical',  # 类别模式
                                     batch_size=BATCH_SIZE)  # 批量大小
# 训练集、验证集和测试集样本数量
nb_train_samples = train_generator.samples  # 训练集样本数量
nb_validation_samples = validation_generator.samples  # 验证集样本数量
nb_test_samples = test_generator.samples  # 测试集样本数量
classes = list(train_generator.class_indices.keys())  # 类别列表
print('Classes: '+str(classes))  # 输出类别列表
num_classes  = len(classes)  # 类别数量

Classes: ['acai', 'cupuacu', 'graviola', 'guarana', 'pupunha', 'tucuma']

# 通过创建的生成器可视化数据集中的一些示例
plt.figure(figsize=(15,15))
for i in range(9):
    # 生成子图
    plt.subplot(330 + 1 + i)
    batch = (train_generator.next()[0]+1)/2*255  # 从训练集生成一个批次，并将像素值缩放到0到255之间
    image = batch[0].astype('uint8')  # 将批次中的第一张图像转换为uint8类型
    plt.imshow(image)  # 显示图像
plt.show()  # 展示图像
base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(im_shape[0], im_shape[1], 3))
x = base_model.output  # 获取基础模型的输出
x = Flatten()(x)  # 将输出展平
x = Dense(100, activation='relu')(x)  # 添加全连接层
predictions = Dense(num_classes, activation='softmax', kernel_initializer='random_uniform')(x)  # 添加输出层
model = Model(inputs=base_model.input, outputs=predictions)  # 创建模型
# 冻结预训练层
base_model.trainable = Faloptimizer = Adam()  # 使用Adam优化器
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])  # 编译模型，设置损失函数和评估指标
epochs = 200  # 训练周期数
# 保存最佳模型
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='model.h5',
        monitor='val_loss', save_best_only=True, verbose=1),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1)]
# 训练模型
history = model.fit(
        train_generator,  #练数据生成器
        steps_per_epoch=nb_train_samples // BATCH_SIZE,  # 每个训练周期的步数
        epochs=epochs,  # 训练周期数
        callbacks=callbacks_list,  # 回调函数列表
        validation_data=validation_generator,  # 验证数据生成器
        verbose=1,  # 日志显示模式
        validation_steps=nb_validation_samples // BATCH_SIZE)  # 验证步数
# 绘制训练和验证损失、准确率曲线
import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict['loss']  # 训练损失
val_loss_values = history_dict['val_loss']  # 验证损失
epochs_x = range(1, len(loss_values) + 1)
plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
plt.plot(epochs_x, loss_values, 'bo', label='Training loss')
plt.plot(epochs_x, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation Loss and Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
#plt.legend()
plt.subplot(2,1,2)
acc_values = history_dict['accuracy']  # 训练准确率
val_acc_values = history_dict['val_accuracy']  # 验证准确率
plt.plot(epochs_x, acc_values, 'bo', label='Training acc')
plt.plot(epochs_x, val_acc_values, 'b', label='Validation acc')
#plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()
from tensorflow.keras.models import load_model
# 加载最佳保存的模型
model = load_model('model.h5')
# 使用验证数据集
score = model.evaluate_generator(validation_generator)
print('Val loss:', score[0])  # 打印验证损失
print('Val accuracy:', score[1])  # 打印验证准确率
# 使用测试数据集
score = model.evaluate_generator(test_generator)
print('Test loss:', score[0])  # 打印测试损失
print('Test accuracy:', score[1])  # 打印测试准确率
import itertools
# 绘制混淆矩阵。设置 Normalize = True/False
def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    此函数打印并绘制混淆矩阵。
    可以通过设置 `normalize=True` 进行归一化。
    """
    plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化混淆矩阵
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):          plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')  # y轴标签为真实标签
    plt.xlabel('Predicted label')  # x轴标签为预测标签
# 导入分类报告
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# 混淆矩阵和分类报告
Y_pred = model.predict_generator(test_generator)#, nb_test_samples // BATCH_SIZE, workers=1)
y_pred = np.argmax(Y_pred, axis=1)
target_names = classes

# 混淆矩阵
cm = confusion_matrix(test_generator.classes, y_pred)
plot_confusion_matrix(cm, target_names, normalize=False, title='Confusion Matrix')  # 绘制混淆矩阵
print('Classification Report')
print(classification_report(test_generator.classes, y_pred, target_names=target_names))  # 打印分类报告
