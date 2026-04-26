import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# 1. 导入 Lasso 回归模型
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# --- 1. 加载数据 ---
housing = fetch_california_housing()
X = housing.data
y = housing.target
feature_names = housing.feature_names

# --- 2. 数据预处理 ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- 3. 构建与训练模型 ---
# 2. 实例化 Lasso 模型
# alpha 是正则化强度，值越大，模型越简单，更多的特征系数会被压缩为0
# max_iter 增加最大迭代次数，确保模型能够收敛
lasso = Lasso(alpha=1.0, max_iter=10000)

# 训练模型
lasso.fit(X_train, y_train)

# --- 4. 预测与评估 ---
y_predict = lasso.predict(X_test)

# 打印模型的权重(系数)和偏置(截距)
print("\n--- 模型参数 ---")
print(f"权重 (Coefficients): {lasso.coef_}")
print(f"偏置 (Intercept): {lasso.intercept_}")

# 评估模型效果
mse = mean_squared_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)
print(f"\n--- 评估指标 ---")
print(f"均方误差 (MSE): {mse:.2f}")
print(f"R² 得分 (R2 Score): {r2:.2f}")

# --- 5. 可视化结果 ---
plt.figure()
x = [i for i in range(len(y_predict))]
plt.plot(x, y_test)
plt.plot(x, y_predict)
plt.legend(['真实值', '预测值'])
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.title('加州房价走势真实与预测值 (Lasso回归)')
plt.show()