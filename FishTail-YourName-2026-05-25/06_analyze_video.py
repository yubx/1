import deeplabcut

config_path = r'C:\Users\Administrator\Desktop\FishTailProject\FishTail-YourName-2024-11-25\config.yaml'
video_path = r'C:\Users\Administrator\Desktop\1.mp4'

# 分析视频
deeplabcut.analyze_videos(
    config_path,
    [video_path],
    shuffle=1,
    save_as_csv=True,           # 保存为CSV格式
    destfolder=r'C:\Users\Administrator\Desktop\FishTailProject\results'
)

# 创建带标注的视频（可视化结果）
deeplabcut.create_labeled_video(
    config_path,
    [video_path],
    shuffle=1,
    save_frames=False,
    destfolder=r'C:\Users\Administrator\Desktop\FishTailProject\results'
)

# 创建轨迹图
deeplabcut.plot_trajectories(
    config_path,
    [video_path],
    shuffle=1,
    destfolder=r'C:\Users\Administrator\Desktop\FishTailProject\results'
)

print("视频分析完成！结果保存在 results 文件夹")