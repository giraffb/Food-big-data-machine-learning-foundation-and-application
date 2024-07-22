import gpflow
from gpflow.utilities import print_summary
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

# 加载数据集
file_path = 'CrabAgePrediction.csv'
crab_age_data = pd.read_csv(file_path)
print(crab_age_data)
# 数据预处理
label_encoder = LabelEncoder()
crab_age_data['Sex'] = label_encoder.fit_transform(crab_age_data['Sex'])

X = crab_age_data.drop('Age', axis=1)
y = crab_age_data['Age']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# 确保 y_train 是 float64 类型
y_train = y_train.astype(np.float64)

# 设置高斯过程回归（GPR）模型
kernel = gpflow.kernels.SquaredExponential()
model = gpflow.models.GPR(data=(X_train, y_train.values.reshape(-1, 1)), kernel=kernel, mean_function=None)

# 优化模型
optimizer = gpflow.optimizers.Scipy()
optimizer.minimize(model.training_loss, variables=model.trainable_variables, options=dict(maxiter=100))

# 打印模型摘要
print_summary(model)
y_pred, y_var = model.predict_f(X_test)

# 打印真实的年龄和预测的年龄
for real_age, pred_age in zip(y_test.values, y_pred):
    print(f"真实年龄：{real_age}，预测年龄：{pred_age[0]}")

from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'均方误差: {mse}')
print(f'R^2 分数: {r2}')

# 示例新实例
new_instance = {
    'Sex': 'F',
    'Length': 1.0,
    'Diameter': 0.8,
    'Height': 0.3,
    'Weight': 10.0,
    'Shucked Weight': 4.0,
    'Viscera Weight': 1.5,
    'Shell Weight': 2.0
}

# 将新实例转换为 DataFrame
new_instance_df = pd.DataFrame([new_instance])

# 编码 'Sex' 列
new_instance_df['Sex'] = label_encoder.transform(new_instance_df['Sex'])

# 缩放特征
new_instance_scaled = scaler.transform(new_instance_df)

# 转换为 float64 以兼容 GPflow
new_instance_scaled = new_instance_scaled.astype(np.float64)

# 使用训练好的模型进行预测
y_pred, y_var = model.predict_f(new_instance_scaled)

print(f'预测年龄: {y_pred.numpy()[0][0]}')
print(f'预测方差: {y_var.numpy()[0][0]}')