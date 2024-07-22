import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")  # 忽略警告
import re  # 正则表达式库
from bs4 import BeautifulSoup  # 用于HTML和XML解析的库
from tqdm import tqdm  # 进度条显示库
from nltk.stem import WordNetLemmatizer  # 词形还原库
from sklearn.model_selection import train_test_split  # 数据集划分库
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix  # 分类模型评价指标
from keras.preprocessing.text import Tokenizer  # 文本预处理库
from keras.preprocessing.sequence import pad_sequences  # 序列填充库
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten  # 神经网络层库
from keras.layers import Bidirectional, GlobalMaxPool1D  # 神经网络层库
from keras.models import Model, Sequential  # 神经网络模型库
from keras.layers import Convolution1D  # 1D卷积层库
from keras import initializers, regularizers, constraints, optimizers, layers  # 神经网络初始化、正则化、约束、优化器库
# 从CSV文件中读取数据
df = pd.read_csv("/kaggle/input/amazon-fine-food-reviews/Reviews.csv")
# 过滤掉评分为3的评论
df_filtered = df[df["Score"]!=3]
# 将评分转换为二元分类问题：评分大于3的为正面情感（1），小于等于3的为负面情感（0）
df_filtered["Score"] = df_filtered["Score"].apply(lambda x: 1 if x > 3 else 0)
# 按照ProductId对数据进行排序
sorted_data = df_filtered.sort_values('ProductId', kind='quicksort', na_position='last')
# 去除重复的评论
final_df = sorted_data.drop_duplicates(subset={"UserId", "ProfileName", "Time", "Text"}, keep='first', inplace=False)
# 去除HelpfulnessNumerator大于HelpfulnessDenominator的异常数据
final_df = final_df[final_df["HelpfulnessNumerator"] <= final_df["HelpfulnessDenominator"]]
# 统计正负面情感的评论数量
final_df['Score'].value_counts()
# 绘制情感分布的条形图
plt.figure(figsize=(10,7))
sns.countplot(final_df['Score'])
plt.title("Bar plot of sentiments")
# 定义函数对评论文本进行解缩、预处理
def decontract(text):
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    return text
# 自定义停用词集合
stop_words= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"])
# 定义词形还原器
lemmatizer = WordNetLemmatizer()
# 定义预处理函数，对评论文本进行多项操作
def preprocess_text(review):
    review = re.sub(r"http\S+", "", review)             # 移除网站链接
    review = BeautifulSoup(review, 'lxml').get_text()   # 移除HTML标签
    review = decontract(review)                         # 解缩
    review = re.sub("\S*\d\S*", "", review).strip()     # 移除包含数字的单词
    review = re.sub('[^A-Za-z]+', ' ', review)          # 移除非单词字符
    review = review.lower()                             # 转换为小写
    review = [word for word in review.split(" ") if not word in stop_words] # 移除停用词
    review = [lemmatizer.lemmatize(token, "v") for token in review] # 词形还原
    review = " ".join(review)
    review.strip()
    return review
# 对final_df数据集中的评论文本应用预处理函数
final_df['Text'] = final_df['Text'].apply(lambda x: preprocess_text(x))
# 将数据集划分为训练集和测试集
train_df, test_df = train_test_split(final_df, test_size=0.2, random_state=42)
# 打印训练集和测试集的大小
print("Training data size : ", train_df.shape)
print("Test data size : ", test_df.shape)
# 设定要保留的前top_words个常用词的数量
top_words = 6000
# 使用Tokenizer对训练集中的文本进行标记化，并仅保留前top_words个常用词
tokenizer = Tokenizer(num_words=top_words)
tokenizer.fit_on_texts(train_df['Text'])
list_tokenized_train = tokenizer.texts_to_sequences(train_df['Text'])
# 设定评论文本的最大长度为130个词，并将训练集的文本序列填充或截断到相同长度
max_review_length = 130
X_train = pad_sequences(list_tokenized_train, maxlen=max_review_length)
y_train = train_df['Score']
# 设定词嵌入向量的长度为32
embedding_vecor_length = 32
# 构建基于LSTM的神经网络模型
model = Sequential()
model.add(Embedding(top_words+1, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))  # 添加LSTM层
model.add(Dense(1, activation='sigmoid'))  # 添加输出层，使用sigmoid激活函数进行二分类
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # 编译模型，使用二分类交叉熵损失函数和Adam优化器
model.summary()  # 打印模型结构信息
# 训练模型
model.fit(X_train, y_train, nb_epoch=3, batch_size=64, validation_split=0.2)
# 对测试集中的文本进行标记化，并使用训练集的tokenizer将其转换为序列
list_tokenized_test = tokenizer.texts_to_sequences(test_df['Text'])
X_test = pad_sequences(list_tokenized_test, maxlen=max_review_length)
y_test = test_df['Score']
# 使用训练好的模型进行预测
prediction = model.predict(X_test)
y_pred = (prediction > 0.5)  # 将概率值大于0.5的预测为正面情感
# 输出模型的准确率、F1分数和混淆矩阵
print("Accuracy of the model : ", accuracy_score(y_pred, y_test))
print('F1-score: ', f1_score(y_pred, y_test))
print('Confusion matrix:')
