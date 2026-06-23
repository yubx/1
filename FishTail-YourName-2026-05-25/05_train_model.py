import deeplabcut

config_path = r'C:\Users\Administrator\Desktop\FishTailProject\FishTail-YourName-2024-11-25\config.yaml'

# 训练神经网络
# 如果使用GPU: displayiters=100, saveiters=500, maxiters=100000
# 如果使用CPU: maxiters=30000 (减少训练轮数)
deeplabcut.train_network(
    config_path,
    shuffle=1,
    trainingsetindex=0,
    max_snapshots_to_keep=5,  # 保留最近5个模型
    displayiters=100,          # 每100次迭代显示损失
    saveiters=500,             # 每500次迭代保存模型
    maxiters=100000            # 最大训练轮数
)

# 评估模型性能
deeplabcut.evaluate_network(config_path, Shuffles=[1])

print("模型训练和评估完成")