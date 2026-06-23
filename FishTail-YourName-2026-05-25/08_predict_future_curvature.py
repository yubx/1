import pandas as pd
import numpy as np
import joblib
import deeplabcut
import warnings
warnings.filterwarnings('ignore')

def predict_future_curvature(video_path, model_path, config_path, fps=30, window=10):
    """
    对新视频进行弯曲度预测
    
    参数:
    - video_path: 新视频路径
    - model_path: 训练好的预测模型路径
    - config_path: DLC配置文件路径
    - fps: 视频帧率
    - window: 用于预测的历史帧数
    """
    
    print(f"分析视频: {video_path}")
    
    # 1. 分析新视频
    deeplabcut.analyze_videos(
        config_path,
        [video_path],
        shuffle=1,
        save_as_csv=True
    )
    
    # 2. 读取分析结果
    import glob
    csv_files = glob.glob(f"{video_path.replace('.mp4', '')}*.csv")
    if not csv_files:
        print("未找到分析结果文件")
        return
    
    data = pd.read_csv(csv_files[0])
    
    # 3. 提取坐标并计算角度
    def extract_coordinates(data, bodypart):
        x_cols = [col for col in data.columns if bodypart in col and 'x' in col.lower()]
        y_cols = [col for col in data.columns if bodypart in col and 'y' in col.lower()]
        if len(x_cols) == 0 or len(y_cols) == 0:
            return None, None
        return data[x_cols[0]].values, data[y_cols[0]].values
    
    def calculate_angle(p1, p2, p3):
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180 / np.pi
        return angle
    
    tail_base_x, tail_base_y = extract_coordinates(data, 'Tail_base')
    tail_mid_x, tail_mid_y = extract_coordinates(data, 'Tail_mid')
    tail_tip_x, tail_tip_y = extract_coordinates(data, 'Tail_tip')
    
    angles = []
    for i in range(len(tail_base_x)):
        if all(pd.notna([tail_base_x[i], tail_mid_x[i], tail_tip_x[i]])):
            p1 = (tail_base_x[i], tail_base_y[i])
            p2 = (tail_mid_x[i], tail_mid_y[i])
            p3 = (tail_tip_x[i], tail_tip_y[i])
            angle = calculate_angle(p1, p2, p3)
            angles.append(angle)
        else:
            angles.append(np.nan)
    
    angles = np.array(angles)
    
    # 4. 使用模型预测
    model = joblib.load(model_path)
    
    predictions = []
    for i in range(window, len(angles) - 1):
        if not any(np.isnan(angles[i-window:i])):
            X_pred = angles[i-window:i].reshape(1, -1)
            pred_angle = model.predict(X_pred)[0]
            predictions.append({
                'current_frame': i,
                'current_angle': angles[i],
                'predicted_next_angle': pred_angle,
                'error': pred_angle - angles[i] if i < len(angles) else None
            })
    
    # 5. 输出结果
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(r'C:\Users\Administrator\Desktop\FishTailProject\predictions.csv', index=False)
    
    print(f"\n===== 预测结果 =====")
    print(f"共生成 {len(pred_df)} 个预测")
    print(f"平均预测误差: {pred_df['error'].mean():.2f}°")
    print(f"预测结果已保存到 predictions.csv")
    
    return pred_df

# 使用示例
if __name__ == "__main__":
    # 配置路径
    config_path = r'C:\Users\Administrator\Desktop\FishTailProject\FishTail-YourName-2024-11-25\config.yaml'
    model_path = r'C:\Users\Administrator\Desktop\FishTailProject\curvature_predictor.pkl'
    new_video = r'C:\Users\Administrator\Desktop\new_video.mp4'  # 替换为你的新视频
    
    # 进行预测
    results = predict_future_curvature(new_video, model_path, config_path)