import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor

try:
    from lightgbm import LGBMRegressor
    lightgbm_available = True
except ImportError:
    lightgbm_available = False
    print("LightGBM is not installed. You can install it using 'pip install lightgbm'.")

try:
    from catboost import CatBoostRegressor
    catboost_available = True
except ImportError:
    catboost_available = False
    print("CatBoost is not installed. You can install it using 'pip install catboost'.")

# 读取数据
nut = pd.read_csv('ABBREV.csv')

# 数据清洗
nut_1 = nut.drop(['GmWt_1', 'GmWt_Desc1', 'GmWt_2', 'GmWt_Desc2', 'Refuse_Pct'], axis=1)
nut_1 = nut_1.dropna()

# 绘制分布图
columns = nut_1.drop(['index', 'NDB_No', 'Shrt_Desc', 'Energ_Kcal'], axis=1)
# fig = go.Figure()
# for column in columns:
#     fig.add_trace(go.Box(y=columns[column], name=column))
# fig.update_layout(
#     title="Distribution of food types across each Nutrient",
#     yaxis_title="Values (each in its unit)",
#     xaxis_title="Nutrients"
# )
# fig.show()

# 计算每个变量的四分位数
data = columns
quartiles = data.quantile([0.25, 0.5, 0.75])

# 根据四分位数对行值进行分类的函数
def classify_row(row, quartiles):
    labels = []
    for column in row.index:
        q1 = quartiles.loc[0.25, column]
        q3 = quartiles.loc[0.75, column]
        if row[column] < q1:
            labels.append("Low")
        elif q1 <= row[column] <= q3:
            labels.append("Medium")
        else:
            labels.append("High")
    return pd.Series(labels, index=row.index)

# 对每列的行进行分类
classifications = data.apply(classify_row, axis=1, args=(quartiles,))
classifications = pd.merge(nut_1['Shrt_Desc'], classifications, left_index=True, right_index=True)

# 定义函数获取特定分类的食品类型
def food_types_cat(nutrient, cat):
    z = pd.DataFrame(classifications[['Shrt_Desc', nutrient]][classifications[nutrient] == cat])
    return z

# 示例调用
food_types_cat('Cholestrl_(mg)', 'High')

# 准备训练数据
nut_measure = nut[['Shrt_Desc', 'Water_(g)', 'Energ_Kcal',
                   'Protein_(g)', 'Lipid_Tot_(g)', 'Ash_(g)', 'Carbohydrt_(g)',
                   'Fiber_TD_(g)', 'Sugar_Tot_(g)']]
nut_measure = nut_measure.dropna()

# 特征和目标变量
y = nut_measure['Energ_Kcal']
X = nut_measure.drop(columns=['Energ_Kcal', 'Shrt_Desc'])

# 数据集划分
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# 模型训练和评估
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

# 交叉验证
def evaluate_model(model, X, y):
    scores = -1 * cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    return scores.mean()

# XGBoost
xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
xgb_model = train_model(xgb_model, X_train, y_train)
xgb_mae = evaluate_model(xgb_model, X, y)

# LightGBM
if lightgbm_available:
    lgb_model = LGBMRegressor(n_estimators=1000, learning_rate=0.05)
    lgb_model = train_model(lgb_model, X_train, y_train)
    lgb_mae = evaluate_model(lgb_model, X, y)
else:
    lgb_mae = None

# CatBoost
if catboost_available:
    cat_model = CatBoostRegressor(iterations=1000, learning_rate=0.05, verbose=0)
    cat_model = train_model(cat_model, X_train, y_train)
    cat_mae = evaluate_model(cat_model, X, y)
else:
    cat_mae = None

# 梯度提升决策树 (GBDT)
gbdt_model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05)
gbdt_model = train_model(gbdt_model, X_train, y_train)
gbdt_mae = evaluate_model(gbdt_model, X, y)

# 打印结果
print(f"XGBoost MAE: {xgb_mae}")
if lgb_mae is not None:
    print(f"LightGBM MAE: {lgb_mae}")
if cat_mae is not None:
    print(f"CatBoost MAE: {cat_mae}")
print(f"GBDT MAE: {gbdt_mae}")
