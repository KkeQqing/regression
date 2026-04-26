import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso  # Lasso 回归
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes

# --- 1. 加载数据---
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names

# --- 2. 数据预处理 ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- 3. 构建 Lasso 回归模型 ---
# alpha 越大，正则化越强，越多特征权重变 0
lr = Lasso(alpha=0.1, random_state=1)

# 训练模型
lr.fit(X_train, y_train)

# --- 4. 预测与评估 ---
y_predict = lr.predict(X_test)

# 打印模型参数
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

# 模型评估
mse = mean_squared_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)
print(f"\n--- 模型评估 ---")
print(f"均方误差 (MSE): {mse:.2f}")
print(f"R² 得分: {r2:.2f}")

# --- 5. 可视化 ---
plt.figure(figsize=(12, 6))
x = list(range(len(y_predict)))

plt.plot(x, y_test, label='真实值')
plt.plot(x, y_predict, label='预测值')
plt.legend()

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.title('Lasso 回归预测结果')
plt.xlabel('样本序号')
plt.ylabel('目标值')

plt.show()