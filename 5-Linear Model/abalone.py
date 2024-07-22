
import pandas as pd


import numpy as np


from sklearn.model_selection import train_test_split


from sklearn.preprocessing import StandardScaler


from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression


from sklearn.metrics import mean_squared_error, r2_score


import matplotlib.pyplot as plt


from sklearn.cross_decomposition import PLSRegression


data = pd.read_csv('abalone.csv')

data = pd.get_dummies(data, columns=['Sex'], drop_first=True)

X = data.drop(columns='Rings')

y = data['Rings']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)

mse_linear = mean_squared_error(y_test, y_pred_linear)

r2_linear = r2_score(y_test, y_pred_linear)

print(f'线性回归 - MSE: {mse_linear}, R^2: {r2_linear}')

ridge_model = Ridge(alpha=1.0)

ridge_model.fit(X_train, y_train)

y_pred_ridge = ridge_model.predict(X_test)

mse_ridge = mean_squared_error(y_test, y_pred_ridge)

r2_ridge = r2_score(y_test, y_pred_ridge)

print(f'岭回归 - MSE: {mse_ridge}, R^2: {r2_ridge}')

lasso_model = Lasso(alpha=0.1)

lasso_model.fit(X_train, y_train)

y_pred_lasso = lasso_model.predict(X_test)

mse_lasso = mean_squared_error(y_test, y_pred_lasso)

r2_lasso = r2_score(y_test, y_pred_lasso)

print(f'Lasso回归 - MSE: {mse_lasso}, R^2: {r2_lasso}')

elastic_model = ElasticNet(alpha=0.1, l1_ratio=0.5)

elastic_model.fit(X_train, y_train)

y_pred_elastic = elastic_model.predict(X_test)

mse_elastic = mean_squared_error(y_test, y_pred_elastic)

r2_elastic = r2_score(y_test, y_pred_elastic)

print(f'弹性网络回归 - MSE: {mse_elastic}, R^2: {r2_elastic}')

pls_model = PLSRegression(n_components=2)

pls_model.fit(X_train, y_train)

y_pred_pls = pls_model.predict(X_test)

mse_pls = mean_squared_error(y_test, y_pred_pls)

r2_pls = r2_score(y_test, y_pred_pls)

print(f'偏最小二乘回归 - MSE: {mse_pls}, R^2: {r2_pls}')

models = {

"线性回归": linear_model,

"岭回归": ridge_model,

"Lasso回归": lasso_model,

"弹性网络回归": elastic_model,

"偏最小二乘回归": pls_model
}
mse_results = {}
r2_results = {}
y_preds = {}

for name, model in models.items():
	    y_pred = model.predict(X_test)
	    y_preds[name] = y_pred
	    mse_results[name] = mean_squared_error(y_test, y_pred)
	    r2_results[name] = r2_score(y_test, y_pred)

print('线性回归权重：', linear_model.coef_)
print('Lasso回归权重：', lasso_model.coef_)

plt.figure(figsize=(15, 10))

for i, (name, y_pred) in enumerate(y_preds.items(), 1):

    plt.subplot(2, 3, i)

    plt.scatter(y_test, y_pred, alpha=0.3)

    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)

    plt.title(f'{name}')

    plt.xlabel('实际值')

    plt.ylabel('预测值')

plt.tight_layout()

plt.show()


plt.figure(figsize=(10, 5))

plt.bar(mse_results.keys(), mse_results.values(), color='skyblue')

plt.ylabel('均方误差 (MSE)')

plt.title('MSE对比')

plt.show()

plt.figure(figsize=(10, 5))
plt.bar(r2_results.keys(), r2_results.values(), color='lightgreen')
plt.ylabel('决定系数 (R方)')
plt.title('R方对比')
plt.show()
