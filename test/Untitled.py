#!/usr/bin/env python
"""
run_system.py - 矿洞内部网格可视化系统（逐帧生成+实时可视化+最终合并）
功能：
1. 逐帧读取PCD文件，每个文件独立生成Mesh
2. 实时可视化：每生成一个Mesh立即显示在窗口中（动态添加）
3. 所有帧处理完毕后，将所有Mesh合并成一个完整网格并保存
4. 支持保存每帧独立网格
"""
import os
import glob
import numpy as np
import open3d as o3d
import time
import argparse
from enum import Enum
from typing import List, Optional, Tuple

# ==================== 配置参数 ====================
PCD_FOLDER = r"C:\Users\Administrator\Desktop\test\test"
OUTPUT_MERGED_MESH = "./merged_mesh.ply"        # 合并后的网格
OUTPUT_SINGLE_PREFIX = "./mesh_frame_"          # 每帧独立网格保存前缀

# 点云处理参数
VOXEL_DOWNSAMPLE = 0.08
OUTLIER_NB = 30
OUTLIER_STD = 2.0

# 网格生成参数
POISSON_DEPTH = 10
MESH_SIMPLIFY_TARGET = 80000
MESH_SMOOTH_ITER = 3

# 运行模式
SAVE_SINGLE_MESHES = True        # 是否保存每帧的独立网格
SAVE_MERGED_MESH = True          # 是否保存合并后的网格
LIVE_VISUALIZATION = True        # 是否实时显示每个生成的Mesh

# 可视化参数
DEFAULT_VIEW_MODE = "solid"
SHOW_COORDINATES = True
SHOW_BOUNDING_BOX = True
# =================================================


class ViewMode(Enum):
    SOLID = "solid"
    WIREFRAME = "wireframe"
    HYBRID = "hybrid"


class MeshGenerator:
    """网格生成器（单文件处理，不融合）"""
    
    def __init__(self):
        self.voxel_size = VOXEL_DOWNSAMPLE
        
    def preprocess_pointcloud(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """预处理：下采样 + 离群点去除"""
        if len(pcd.points) == 0:
            return pcd
        pcd = pcd.voxel_down_sample(self.voxel_size)
        if len(pcd.points) > 100:
            _, ind = pcd.remove_statistical_outlier(nb_neighbors=OUTLIER_NB, std_ratio=OUTLIER_STD)
            pcd = pcd.select_by_index(ind)
        return pcd
    
    def generate_mesh_from_pcd(self, pcd: o3d.geometry.PointCloud) -> Optional[o3d.geometry.TriangleMesh]:
        """从单帧点云生成网格"""
        if pcd is None or len(pcd.points) < 100:
            return None
        
        # 进一步下采样加速泊松重建
        pcd_for_mesh = pcd.voxel_down_sample(0.1)
        print(f"      参与重建点数: {len(pcd_for_mesh.points):,}")
        
        # 法向量估计
        pcd_for_mesh.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30)
        )
        pcd_for_mesh.orient_normals_consistent_tangent_plane(k=50)
        
        try:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd_for_mesh, depth=POISSON_DEPTH, width=0, scale=1.1, linear_fit=True
            )
            densities = np.asarray(densities)
            if len(densities) > 0:
                threshold = np.percentile(densities, 10)
                mesh.remove_vertices_by_mask(densities < threshold)
            
            mesh = mesh.remove_duplicated_vertices()
            mesh = mesh.remove_duplicated_triangles()
            mesh = mesh.remove_degenerate_triangles()
            mesh = mesh.remove_non_manifold_edges()
            
            if MESH_SMOOTH_ITER > 0:
                mesh = mesh.filter_smooth_taubin(number_of_iterations=MESH_SMOOTH_ITER)
            if len(mesh.triangles) > MESH_SIMPLIFY_TARGET:
                mesh = mesh.simplify_quadric_decimation(MESH_SIMPLIFY_TARGET)
            
            mesh.compute_vertex_normals()
            return mesh
        except Exception as e:
            print(f"      网格生成失败: {e}")
            return None
    
    def process_single_file(self, file_path: str, frame_id: int, save_path: str = None) -> Optional[o3d.geometry.TriangleMesh]:
        """处理单个PCD文件：读取 -> 预处理 -> 生成网格 -> 可选保存"""
        print(f"\n  处理第 {frame_id} 帧: {os.path.basename(file_path)}")
        start = time.time()
        
        try:
            pcd = o3d.io.read_point_cloud(file_path)
        except Exception as e:
            print(f"    读取失败: {e}")
            return None
        
        if len(pcd.points) < 100:
            print(f"    点数不足 ({len(pcd.points)}), 跳过")
            return None
        
        pcd = self.preprocess_pointcloud(pcd)
        pts = np.asarray(pcd.points)
        print(f"    预处理后点数: {len(pts):,}")
        print(f"    范围 X:[{pts[:,0].min():.1f},{pts[:,0].max():.1f}] Y:[{pts[:,1].min():.1f},{pts[:,1].max():.1f}]")
        
        mesh = self.generate_mesh_from_pcd(pcd)
        if mesh is None:
            return None
        
        if save_path and SAVE_SINGLE_MESHES:
            o3d.io.write_triangle_mesh(save_path, mesh)
            print(f"    已保存独立网格: {save_path}")
        
        elapsed = time.time() - start
        print(f"    完成，耗时 {elapsed:.1f}s，顶点 {len(mesh.vertices):,}，三角形 {len(mesh.triangles):,}")
        return mesh


class LiveVisualizer:
    """实时可视化器：支持动态添加几何体"""
    def __init__(self, view_mode="solid", show_coords=True, show_bbox=True):
        self.vis = None
        self.view_mode = view_mode
        self.show_coords = show_coords
        self.show_bbox = show_bbox
        self.meshes_added = 0
        self.bounds_updated = False
        
    def start(self):
        """创建窗口并初始化"""
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="矿洞网格实时可视化", width=1280, height=720)
        
        # 设置渲染选项
        opt = self.vis.get_render_option()
        opt.background_color = np.array([0.05, 0.05, 0.1])
        opt.mesh_show_back_face = True
        opt.line_width = 1.5
        
        # 初始添加坐标轴 
        if self.show_coords:
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
            self.vis.add_geometry(coord)
        
        # 设置相机视角
        ctrl = self.vis.get_view_control()
        ctrl.set_front([0.5, -0.5, -0.5])
        ctrl.set_lookat([0, 0, 2])
        ctrl.set_up([0, 0, 1])
        ctrl.set_zoom(0.6)
        
        self.vis.poll_events()
        self.vis.update_renderer()
        
    def add_mesh(self, mesh: o3d.geometry.TriangleMesh):
        """添加一个网格到窗口并刷新"""
        if mesh is None or self.vis is None:
            return
        
        # 为网格上色（根据高度渐变）
        vertices = np.asarray(mesh.vertices)
        if len(vertices) > 0:
            z_min, z_max = vertices[:,2].min(), vertices[:,2].max()
            if z_max - z_min > 0:
                colors = np.zeros((len(vertices), 3))
                for i, v in enumerate(vertices):
                    t = (v[2] - z_min) / (z_max - z_min)
                    colors[i] = [t, 0.3, 1-t]
                mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            else:
                mesh.paint_uniform_color([0.7, 0.7, 0.7])
        
        # 根据显示模式添加
        if self.view_mode == "solid":
            self.vis.add_geometry(mesh)
        elif self.view_mode == "wireframe":
            wireframe = self._create_wireframe(mesh)
            if wireframe:
                self.vis.add_geometry(wireframe)
        else:  # hybrid
            self.vis.add_geometry(mesh)
            wireframe = self._create_wireframe(mesh)
            if wireframe:
                wireframe.paint_uniform_color([0.5, 0.5, 0.5])
                self.vis.add_geometry(wireframe)
        
         
        if self.show_bbox and not self.bounds_updated:
            # 简单起见，这里只添加一次固定边界框（可在所有网格添加完后更新）
            pass
        
        self.meshes_added += 1
        
        self.vis.poll_events()
        self.vis.update_renderer()
        time.sleep(0.05)  
        
    def _create_wireframe(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.LineSet:
        triangles = np.asarray(mesh.triangles)
        edges_set = set()
        for tri in triangles:
            edges_set.add(tuple(sorted([tri[0], tri[1]])))
            edges_set.add(tuple(sorted([tri[1], tri[2]])))
            edges_set.add(tuple(sorted([tri[2], tri[0]])))
        edges_array = np.array(list(edges_set))
        wireframe = o3d.geometry.LineSet()
        wireframe.points = mesh.vertices
        wireframe.lines = o3d.utility.Vector2iVector(edges_array)
        wireframe.paint_uniform_color([0.8, 0.8, 0.8])
        return wireframe
    
    def run(self):
        """进入主循环，直到用户关闭窗口"""
        if self.vis:
            self.vis.run()
    
    def close(self):
        if self.vis:
            self.vis.destroy_window()


class MineMeshSystem:
    """矿洞网格系统：逐帧生成Mesh，实时可视化，最后合并"""
    
    def __init__(self, pcd_folder: str):
        self.folder = pcd_folder
        self.generator = MeshGenerator()
        self.meshes = []
        self.merged_mesh = None
        
    def get_sorted_files(self) -> List[str]:
        files = glob.glob(os.path.join(self.folder, "*.pcd")) + glob.glob(os.path.join(self.folder, "*.PCD"))
        files.sort()
        if not files:
            print(f"错误: 在 {self.folder} 中未找到PCD文件")
        return files
    
    def build_and_merge(self, live_view=True, view_mode="solid", show_coords=True, show_bbox=True) -> bool:
        """逐帧生成Mesh，实时可视化，最后合并"""
        print("\n" + "="*60)
        print("逐帧生成Mesh并实时可视化")
        print("="*60)
        
        files = self.get_sorted_files()
        if not files:
            return False
        
        total = len(files)
        print(f"共找到 {total} 个PCD文件")
        
        # 初始化实时可视化器
        live_vis = None
        if live_view:
            live_vis = LiveVisualizer(view_mode, show_coords, show_bbox)
            live_vis.start()
            print("实时可视化窗口已启动，每生成一个Mesh就会显示")
        
        self.meshes = []
        failed_frames = []
        
        for idx, f in enumerate(files):
            save_path = f"{OUTPUT_SINGLE_PREFIX}{idx:04d}.ply" if SAVE_SINGLE_MESHES else None
            mesh = self.generator.process_single_file(f, idx, save_path)
            if mesh is not None:
                self.meshes.append(mesh)
                
                if live_view and live_vis:
                    live_vis.add_mesh(mesh)
            else:
                failed_frames.append(idx)
        
        if not self.meshes:
            print("没有成功生成任何网格")
            if live_view and live_vis:
                live_vis.close()
            return False
        
        print(f"\n成功生成 {len(self.meshes)} 个网格，失败 {len(failed_frames)} 个")
        
        # 如果实时可视化窗口还在，可以选择进入交互模式或关闭后继续
        if live_view and live_vis:
            print("\n所有网格已生成。窗口保持打开")
            print("关闭窗口后将进行网格合并和保存...")
            live_vis.run()  # 阻塞直到用户关闭窗口
            live_vis.close()
        
        # 合并所有网格
        print("\n正在合并所有网格...")
        merge_start = time.time()
        merged = self.meshes[0]
        for i in range(1, len(self.meshes)):
            merged += self.meshes[i]
            if (i+1) % 10 == 0:
                print(f"  已合并 {i+1}/{len(self.meshes)} 个网格")
        
        print("清理合并后的网格...")
        merged = merged.remove_duplicated_vertices()
        merged = merged.remove_duplicated_triangles()
        merged = merged.remove_degenerate_triangles()
        merged = merged.remove_non_manifold_edges()
        merged.compute_vertex_normals()
        
        elapsed = time.time() - merge_start
        print(f"合并完成，耗时 {elapsed:.1f}s")
        print(f"合并后网格: 顶点 {len(merged.vertices):,}, 三角形 {len(merged.triangles):,}")
        
        self.merged_mesh = merged
        
        if SAVE_MERGED_MESH:
            o3d.io.write_triangle_mesh(OUTPUT_MERGED_MESH, merged)
            print(f"合并网格已保存: {OUTPUT_MERGED_MESH}")
        
        return True
    
    def visualize_merged(self, view_mode="solid", show_coords=True, show_bbox=True):
        """单独显示合并后的网格（非实时模式）"""
        if self.merged_mesh is None:
            print("构建合并网格")
            return False
        
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        # 简单着色
        self.merged_mesh.paint_uniform_color([0.7, 0.7, 0.7])
        vis.add_geometry(self.merged_mesh)
        if show_coords:
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
            vis.add_geometry(coord)
        vis.run()
        vis.destroy_window()
        return True


def main():
    global SAVE_SINGLE_MESHES, SAVE_MERGED_MESH, LIVE_VISUALIZATION

    parser = argparse.ArgumentParser(description="矿洞内部网格可视化系统（逐帧生成+实时可视化+合并）")
    parser.add_argument("--mode", choices=["build", "view", "all"], default="all")
    parser.add_argument("--input", type=str, default=PCD_FOLDER, help="PCD文件夹路径")
    parser.add_argument("--save_single", action="store_true", default=SAVE_SINGLE_MESHES,
                        help="是否保存每帧独立网格")
    parser.add_argument("--save_merged", action="store_true", default=SAVE_MERGED_MESH,
                        help="是否保存合并后的网格")
    parser.add_argument("--live_view", action="store_true", default=LIVE_VISUALIZATION,
                        help="是否实时显示每个生成的Mesh")
    parser.add_argument("--view_mode", choices=["solid", "wireframe", "hybrid"], 
                        default=DEFAULT_VIEW_MODE, help="显示模式")
    parser.add_argument("--no_coord", action="store_true", help="不显示坐标轴")
    parser.add_argument("--no_bbox", action="store_true", help="不显示边界框")
    
    args = parser.parse_args()
    
    SAVE_SINGLE_MESHES = args.save_single
    SAVE_MERGED_MESH = args.save_merged
    LIVE_VISUALIZATION = args.live_view
    
    system = MineMeshSystem(args.input)
    
    if args.mode in ["build", "all"]:
        if not system.build_and_merge(
            live_view=LIVE_VISUALIZATION,
            view_mode=args.view_mode,
            show_coords=not args.no_coord,
            show_bbox=not args.no_bbox
        ):
            print("构建失败")
            return
    
    if args.mode in ["view", "all"] and system.merged_mesh is not None:
       
        if args.mode == "view" or not LIVE_VISUALIZATION:
            system.visualize_merged(
                view_mode=args.view_mode,
                show_coords=not args.no_coord,
                show_bbox=not args.no_bbox
            )


if __name__ == "__main__":
    main()