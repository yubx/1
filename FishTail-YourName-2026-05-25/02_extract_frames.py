import deeplabcut

# 替换为你的实际配置文件路径
config_path = r'C:\Users\Administrator\Desktop\FishTailProject\FishTail-YourName-2024-11-25\config.yaml'

# 提取50帧用于标注（可以根据需要调整数量）
deeplabcut.extract_frames(
    config_path,
    mode='automatic',
    algo='kmeans',  # 使用k-means算法选择多样化的帧
    userfeedback=False,
    crop=False
)

print("帧提取完成！现在可以开始标注了")