import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso  # Lasso 回归
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# --- 1. 加载数据 ---
# 使用加州房价数据集，和线性回归那套保持一致
housing = fetch_california_housing()
X = housing.data  # 平均收入、房屋年龄、房间数、卧室数、居住人数、位置等
y = housing.target
feature_names = housing.feature_names

# --- 2. 数据预处理 ---
# 划分训练集和测试集 (70% 训练, 30% 测试)
# random_state=1 保证每次运行结果一致
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
print("X_test:\n ", X_test)
print("X_train:\n ", X_train)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- 3. 构建与训练 Lasso 模型 ---
# alpha 越大，正则化越强，特征权重越容易变成 0（自动特征选择）
lr = Lasso(alpha=0.01, random_state=1)

# 训练模型
lr.fit(X_train, y_train)

# --- 4. 预测与评估 ---
# 在测试集上进行预测
y_predict = lr.predict(X_test)

# 打印模型的权重(系数)和偏置(截距)
print("\n--- Lasso 模型参数 ---")
print(f"权重 (Coefficients): {lr.coef_}")
print(f"偏置 (Intercept): {lr.intercept_}")

# 打印特征选择结果（Lasso 核心功能）
print("\n--- 特征选择（权重=0 表示被舍弃）---")
for name, coef in zip(feature_names, lr.coef_):
    if coef == 0:
        print(f"❌ 特征 {name} 被舍弃")
    else:
        print(f"✅ 特征 {name} 保留，权重：{coef:.4f}")

# 评估模型效果
mse = mean_squared_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)
print(f"\n--- 评估指标 ---")
# 真实值 - 预测值，算出每一次的误差，平方，求平均值
# MSE 越小越好；MSE = 0：预测完全精准，和真实值一模一样
print(f"均方误差 (MSE): {mse:.2f}")
# R方越接近 1 越好，模型比瞎猜强多少
print(f"R² 得分 (R2 Score): {r2:.2f}")

# --- 5. 可视化结果 ---
# 绘制真实房价走势 与预测的房价走势

# 创建画布
plt.figure()

# 绘图-准备数据
x = [i for i in range(len(y_predict))]

# 绘制真实值曲线
plt.plot(x, y_test)

# 绘制预测值曲线
plt.plot(x, y_predict)

# 添加图例
plt.legend(['真实值', '预测值'])

# 修改rc参数, 增加支持中文
plt.rcParams['font.sans-serif'] = 'SimHei'

# 修改rc参数, 增加支持负号
plt.rcParams['axes.unicode_minus'] = False

# 添加标题
plt.title('加州房价走势真实与预测值（Lasso 回归）')

# 展示
plt.show()