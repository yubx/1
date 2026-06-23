import deeplabcut
import os

# 设置路径
video_path = r'C:\Users\Administrator\Desktop\1.mp4'
project_folder = r'C:\Users\Administrator\Desktop\FishTailProject'

# 创建项目（将"YourName"替换为你的名字）
project_name = 'FishTail'
experimenter = 'YourName'

config_path = deeplabcut.create_new_project(
    project_name, 
    experimenter, 
    [video_path], 
    working_directory=project_folder,
    copy_videos=True  # 复制视频到项目文件夹
)

print(f"项目已创建，配置文件路径：{config_path}")

# 修改配置文件，设置关键点
import yaml

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# 设置身体部位关键点（至少3个点用于测量弯曲度）
config['bodyparts'] = [
    'Tail_base',   # 尾巴基部
    'Tail_mid',    # 尾巴中部
    'Tail_tip'     # 尾巴尖端
]

# 设置骨架连接（用于可视化）
config['skeleton'] = [
    ['Tail_base', 'Tail_mid'],
    ['Tail_mid', 'Tail_tip']
]

# 设置每帧中的个体数量
config['numindividuals'] = 1

# 保存修改后的配置
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print("配置文件已更新，关键点设置完成")