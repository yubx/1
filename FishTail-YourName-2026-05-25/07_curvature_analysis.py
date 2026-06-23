import deeplabcut
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 配置路径
config_path = r'C:\Users\Administrator\Desktop\FishTailProject\FishTail-YourName-2024-11-25\config.yaml'
csv_path = r'C:\Users\Administrator\Desktop\FishTailProject\results\1DLC_resnet50_FishTailshuffle1_100000.csv'

# 读取DLC分析结果
print("加载数据...")
data = pd.read_csv(csv_path)

# 提取坐标数据
def extract_coordinates(data, bodypart):
    """提取指定身体部位的坐标"""
    x_cols = [col for col in data.columns if bodypart in col and 'x' in col.lower()]
    y_cols = [col for col in data.columns if bodypart in col and 'y' in col.lower()]
    
    if len(x_cols) == 0 or len(y_cols) == 0:
        return None, None
    
    x = data[x_cols[0]].values
    y = data[y_cols[0]].values
    return x, y

# 提取三个关键点的坐标
tail_base_x, tail_base_y = extract_coordinates(data, 'Tail_base')
tail_mid_x, tail_mid_y = extract_coordinates(data, 'Tail_mid')
tail_tip_x, tail_tip_y = extract_coordinates(data, 'Tail_tip')

if tail_base_x is None:
    print("错误：找不到关键点，请检查CSV文件中的列名")
    print("可用的列名：", [col for col in data.columns if 'tail' in col.lower()])
    exit()

# 计算弯曲角度（使用向量夹角公式）
def calculate_angle(p1, p2, p3):
    """
    计算三点之间的角度
    p1: 起点 (Tail_base)
    p2: 中点 (Tail_mid)  
    p3: 终点 (Tail_tip)
    返回角度（度数）
    """
    # 计算两个向量
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    # 计算夹角
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 避免数值误差
    angle = np.arccos(cos_angle) * 180 / np.pi
    
    return angle

# 计算每帧的弯曲角度
angles = []
for i in range(len(tail_base_x)):
    if pd.notna(tail_base_x[i]) and pd.notna(tail_mid_x[i]) and pd.notna(tail_tip_x[i]):
        p1 = (tail_base_x[i], tail_base_y[i])
        p2 = (tail_mid_x[i], tail_mid_y[i])
        p3 = (tail_tip_x[i], tail_tip_y[i])
        angle = calculate_angle(p1, p2, p3)
        angles.append(angle)
    else:
        angles.append(np.nan)

angles = np.array(angles)

# 计算角速度（角度变化率）
def calculate_angular_velocity(angles, fps=30):
    """计算角速度（度/秒）"""
    dt = 1.0 / fps  # 假设视频30fps
    velocity = np.gradient(angles, dt)
    return velocity

# 假设视频帧率为30fps，你可以根据实际视频调整
fps = 30
angular_velocity = calculate_angular_velocity(angles, fps)

# 创建结果DataFrame
results_df = pd.DataFrame({
    'frame': range(len(angles)),
    'angle_degrees': angles,
    'angular_velocity': angular_velocity
})

# 添加滑动平均平滑（去除噪声）
window_size = 5
results_df['angle_smoothed'] = results_df['angle_degrees'].rolling(window=window_size, center=True).mean()
results_df['velocity_smoothed'] = results_df['angular_velocity'].rolling(window=window_size, center=True).mean()

# 保存结果
results_df.to_csv(r'C:\Users\Administrator\Desktop\FishTailProject\curvature_analysis.csv', index=False)
print(f"弯曲度分析结果已保存，有效数据点：{len(angles[~np.isnan(angles)])}")

# ========== 可视化 ==========
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# 图1：原始角度序列
axes[0].plot(results_df['frame'], results_df['angle_degrees'], 'b-', alpha=0.5, label='原始数据')
axes[0].plot(results_df['frame'], results_df['angle_smoothed'], 'r-', linewidth=2, label='平滑后')
axes[0].set_ylabel('弯曲角度 (度)')
axes[0].set_xlabel('帧数')
axes[0].set_title('鱼尾弯曲角度随时间变化')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 图2：角速度
axes[1].plot(results_df['frame'], results_df['angular_velocity'], 'g-', alpha=0.5, label='原始')
axes[1].plot(results_df['frame'], results_df['velocity_smoothed'], 'm-', linewidth=2, label='平滑后')
axes[1].set_ylabel('角速度 (度/秒)')
axes[1].set_xlabel('帧数')
axes[1].set_title('尾巴摆动速度')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 图3：角度分布直方图
axes[2].hist(results_df['angle_degrees'].dropna(), bins=30, color='skyblue', edgecolor='black')
axes[2].set_xlabel('弯曲角度 (度)')
axes[2].set_ylabel('频次')
axes[2].set_title('角度分布统计')
axes[2].axvline(results_df['angle_degrees'].mean(), color='red', linestyle='--', label=f'均值: {results_df["angle_degrees"].mean():.1f}°')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'C:\Users\Administrator\Desktop\FishTailProject\curvature_visualization.png', dpi=150)
plt.show()

# 打印统计信息
print("\n===== 弯曲度统计 =====")
print(f"平均弯曲角度: {np.nanmean(angles):.2f}°")
print(f"最大弯曲角度: {np.nanmax(angles):.2f}°")
print(f"最小弯曲角度: {np.nanmin(angles):.2f}°")
print(f"角度标准差: {np.nanstd(angles):.2f}°")

# ========== 预测模型：基于历史角度预测未来弯曲度 ==========
def prepare_prediction_data(angles, window_size=10, predict_ahead=1):
    """准备时间序列预测数据"""
    X, y = [], []
    for i in range(window_size, len(angles) - predict_ahead):
        if not any(np.isnan(angles[i-window_size:i+predict_ahead])):
            X.append(angles[i-window_size:i])
            y.append(angles[i+predict_ahead])
    return np.array(X), np.array(y)

# 准备数据
window = 10  # 使用过去10帧预测
predict_ahead = 5  # 预测未来5帧
X, y = prepare_prediction_data(angles, window, predict_ahead)

if len(X) > 100:  # 确保有足够数据
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练随机森林回归模型
    print(f"\n===== 训练预测模型 =====")
    print(f"训练样本数: {len(X_train)}, 测试样本数: {len(X_test)}")
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    # 评估模型
    train_score = rf_model.score(X_train, y_train)
    test_score = rf_model.score(X_test, y_test)
    
    print(f"训练集 R² 分数: {train_score:.4f}")
    print(f"测试集 R² 分数: {test_score:.4f}")
    
    # 特征重要性分析
    feature_importance = rf_model.feature_importances_
    
    # 预测可视化
    y_pred = rf_model.predict(X_test)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, edgecolors='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    plt.xlabel('实际角度 (度)')
    plt.ylabel('预测角度 (度)')
    plt.title(f'随机森林预测模型 (R² = {test_score:.3f})')
    plt.grid(True, alpha=0.3)
    plt.savefig(r'C:\Users\Administrator\Desktop\FishTailProject\prediction_model.png', dpi=150)
    plt.show()
    
    # 保存预测模型（用于后续使用）
    import joblib
    joblib.dump(rf_model, r'C:\Users\Administrator\Desktop\FishTailProject\curvature_predictor.pkl')
    print("预测模型已保存")
else:
    print(f"\n警告：数据不足（{len(X)}个样本），需要更多标注数据来训练预测模型")

print("\n所有分析完成！结果保存在 FishTailProject 文件夹中")