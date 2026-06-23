import deeplabcut

config_path = r'C:\Users\Administrator\Desktop\FishTailProject\FishTail-YourName-2024-11-25\config.yaml'

# 检查标注数据质量
deeplabcut.check_labels(config_path)

# 创建训练数据集（80%训练，20%验证）
deeplabcut.create_training_dataset(
    config_path,
    num_shuffles=1,      # 数据打乱次数
    net_type='resnet_50', # 使用ResNet-50作为骨干网络
    augmenter_type='imgaug'
)

print("训练数据集创建完成")