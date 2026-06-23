import deeplabcut
import os

# 配置
video_path = r'C:\Users\Administrator\Desktop\1.mp4'
project_folder = r'C:\Users\Administrator\Desktop\FishTailProject'
project_name = 'FishTail'
experimenter = 'Researcher'

print("=" * 50)
print("DeepLabCut 项目完整设置")
print("=" * 50)

# 1. 创建项目
print("\n[1/4] 创建项目...")
config_path = deeplabcut.create_new_project(
    project_name,
    experimenter,
    [video_path],
    working_directory=project_folder,
    copy_videos=True
)
print(f"✓ 项目已创建: {config_path}")

# 2. 更新配置文件，设置关键点
print("\n[2/4] 配置关键点...")
import yaml

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# 设置尾巴的三个关键点
config['bodyparts'] = ['Tail_base', 'Tail_mid', 'Tail_tip']
config['skeleton'] = [['Tail_base', 'Tail_mid'], ['Tail_mid', 'Tail_tip']]
config['numindividuals'] = 1

with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
print(f"✓ 关键点已设置: {config['bodyparts']}")

# 3. 提取标注帧
print("\n[3/4] 提取标注帧...")
deeplabcut.extract_frames(
    config_path,
    mode='automatic',
    algo='uniform',  # 均匀采样
    userfeedback=False
)
print("✓ 帧提取完成")

# 4. 验证
print("\n[4/4] 验证项目结构...")
project_dir = os.path.dirname(config_path)
labeled_data_dir = os.path.join(project_dir, 'labeled-data')

if os.path.exists(labeled_data_dir):
    subdirs = [d for d in os.listdir(labeled_data_dir) 
               if os.path.isdir(os.path.join(labeled_data_dir, d))]
    print(f"✓ labeled-data 目录存在")
    for subdir in subdirs:
        img_dir = os.path.join(labeled_data_dir, subdir)
        img_count = len([f for f in os.listdir(img_dir) if f.endswith('.png')])
        print(f"  - {subdir}: {img_count} 张图片")
else:
    print("✗ labeled-data 目录不存在")

print("\n" + "=" * 50)
print("✅ 项目设置完成！")
print(f"配置文件: {config_path}")
print("\n现在可以运行标注命令:")
print(f"deeplabcut.label_frames(r'{config_path}')")
print("=" * 50)