# 导入所需的库
import os  # 导入操作系统相关的库
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split  # 导入数据集划分库
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from tensorflow.keras.applications import ResNet50V2, MobileNetV2  # 导入预训练模型
from tensorflow.keras.applications.resnet import preprocess_input  # 导入图像预处理函数
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # 导入图像数据生成器
# 设置数据集的目录
dir = '../data/a-large-scale-fish-dataset/Fish_Dataset/Fish_Dataset'
# 初始化标签和路径列表
label = []
path = []

# 遍历目录下的文件
for dirname, _, filenames in os.walk(dir):
    for filename in filenames:
        # 如果文件是png格式的图像文件
        if os.path.splitext(filename)[1] == '.png':
            # 如果文件夹名称不是以'GT'结尾
            if dirname.split()[-1] != 'GT':
                # 将文件夹名称作为标签
                label.append(os.path.split(dirname)[1])
                # 将图像路径添加到路径列表中
                path.append(os.path.join(dirname, filename))

# 创建DataFrame存储路径和标签信息
df = pd.DataFrame(columns=['path', 'label'])
df['path'] = path  # 添加路径信息到DataFrame中的path列
df['label'] = label  # 添加标签信息到DataFrame中的label列
df['label'] = df['label'].astype('category')  # 将标签列转换为category类型

# 设置数据集的目录
dir = '../input/a-large-scale-fish-dataset/Fish_Dataset/Fish_Dataset'

# 初始化标签和路径列表
label = []
path = []

# 遍历目录下的文件
for dirname, _, filenames in os.walk(dir):
    for filename in filenames:
        # 如果文件是png格式的图像文件
        if os.path.splitext(filename)[1] == '.png':
            # 如果文件夹名称不是以'GT'结尾
            if dirname.split()[-1] != 'GT':
                # 将文件夹名称作为标签
                label.append(os.path.split(dirname)[1])
                # 将图像路径添加到路径列表中
                path.append(os.path.join(dirname, filename))

# 创建DataFrame存储路径和标签信息
df = pd.DataFrame(columns=['path', 'label'])
df['path'] = path  # 添加路径信息到DataFrame中的path列
df['label'] = label  # 添加标签信息到DataFrame中的label列
df['label'] = df['label'].astype('category')  # 将标签列转换为category类型

# 获取标签的唯一值
labels = df['label'].unique()

# 创建子图布局
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15, 8), constrained_layout=True)
ax = ax.flatten()  # 将二维数组展平为一维数组
j = 0

# 遍历标签的唯一值
for i in labels:
    # 显示每个类别的第一张图像
    ax[j].imshow(plt.imread(df[df['label'] == i].iloc[0, 0]))
    ax[j].set_title(i)  # 设置子图标题为类别名称
    ax[j].axis('off')  # 不显示坐标轴
    j = j + 1

# 创建新的图形
fig = plt.figure(figsize=(15, 8))
# 绘制标签计数的柱状图
sns.countplot(df['label'])
plt.title('Label Count')  # 设置图形标题

# 构造训练集和测试集
X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

plt.show()  # 显示所有图形


# 创建训练集和验证集的图像数据生成器，设置图像预处理函数和验证集比例
trainGen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.3)
testGen = ImageDataGenerator(preprocessing_function= preprocess_input)

# 从DataFrame中创建训练集图像生成器
X_train_img = trainGen.flow_from_dataframe(dataframe=X_train, x_col='path', y_col='label',
                                            class_mode='categorical', subset='training',
                                            color_mode='rgb', batch_size=32)

# 从DataFrame中创建验证集图像生成器
X_val_img = trainGen.flow_from_dataframe(dataframe=X_train, x_col='path', y_col='label',
                                          class_mode='categorical', subset='validation',
                                          color_mode='rgb', batch_size=32)  # 从DataFrame中创建测试集图像生成器

X_test_img = testGen.flow_from_dataframe(dataframe=X_test, x_col='path', y_col='label',
                                          class_mode='categorical', color_mode='rgb',
                                          batch_size=32, shuffle=False)

# 创建子图布局
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15,8))
ax = ax.flatten()  # 将二维数组展平为一维数组
j = 0

# 从测试集生成器中获取图像和标签并显示前6个图像
for _ in range(6):
    img, label = X_test_img.next()
    ax[j].imshow(img[0],)  # 显示图像
    ax[j].set_title(label[0])  # 设置子图标题为标签
    j = j + 1

image_shape = (256, 256, 3)  # 设置图像形状


# 使用MobileNetV2作为预训练模型，去掉顶层，并设置池化方式和输入形状
pre_trained = MobileNetV2(include_top=False, pooling='avg', input_shape=(256, 256, 3))

# 冻结预训练模型的所有层
pre_trained.trainable = False

# 构建模型结构
inp_model = pre_trained.input
x = Dense(128, activation='relu')(pre_trained.output)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
output = Dense(9, activation='softmax')(x)
model = Model(inputs=inp_model, outputs=output)

# 编译模型，指定损失函数、优化器和评估指标
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 设置早停回调函数，防止过拟合
early_stop = EarlyStopping(monitor='val_loss', patience=1)

# 训练模型，并在验证集上进行验证，使用早停回调函数
results = model.fit(X_train_img, epochs=30, validation_data=X_val_img, callbacks=[early_stop])

# 将训练过程的结果转换为DataFrame
result = pd.DataFrame(results.history)

# 绘制训练过程中的准确率和损失变化曲线
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
ax = ax.flatten()
ax[0].plot(result[['accuracy', 'val_accuracy']])
ax[0].set_title("Accuracy")
ax[1].plot(result[['loss', 'val_loss']])
ax[1].set_title("Loss")
plt.show()

pred = model.predict(X_test_img)  # 对测试集进行预测
pred = np.argmax(pred, axis=1)  # 获取预测结果中概率最大的类别索引

pred_df = X_test.copy()  # 复制测试集数据作为预测结果的基础数据
labels = {}  # 创建一个字典，用于存储类别索引和类别名称的对应关系
for l, v in X_test_img.class_indices.items():
    labels.update({v: l})  # 将类别索引和类别名称的对应关系添加到字典中

pred_df['pred'] = pred  # 将预测结果添加到DataFrame中
pred_df['pred'] = pred_df['pred'].apply(lambda x: labels[x])  # 将预测结果中的类别索引转换为类别名称

# 输出模型的准确率
print(f"Accuracy Score: {accuracy_score(pred_df['label'], pred_df['pred'])}")

# 绘制混淆矩阵热力图
sns.heatmap(confusion_matrix(pred_df['label'], pred_df['pred']), annot=True, fmt='2d')
plt.show()