import os
import sys
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from graspnetAPI import GraspGroup

# 假设此脚本文件的路径下有以下目录结构（根据实际情况修改）
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

# ==================== 硬编码配置参数 ====================
CHECKPOINT_PATH = '/home/ncut/graspnet-baseline/pretrain/checkpoint-rs.tar'  # 修改为你的权重路径
NUM_POINT = 10000
NUM_VIEW = 300
COLLISION_THRESH = 0.01
VOXEL_SIZE = 0.01

# 相机内参（使用深度相机参数生成点云，根据实际相机修改）
DEPTH_INTR = {
    "ppx": 319.304,  # cx
    "ppy": 236.915,  # cy
    "fx": 387.897,   # fx
    "fy": 387.897    # fy
}
DEPTH_FACTOR = 1000.0  # 深度因子，根据实际数据调整

# ==================== 网络加载 ====================
def get_net():
    """
    加载训练好的 GraspNet 模型
    """
    net = GraspNet(
        input_feature_dim=0,
        num_view=NUM_VIEW,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    return net

# ==================== 数据处理 ====================
def get_and_process_data(color_path, depth_path, mask_path):
    """
    根据给定的 RGB 图、深度图、掩码图（可以是 文件路径 或 NumPy 数组），生成输入点云及其它必要数据
    """
    # 1. 加载 color（可能是路径，也可能是数组）
    if isinstance(color_path, str):
        color = np.array(Image.open(color_path), dtype=np.float32) / 255.0
    elif isinstance(color_path, np.ndarray):
        color = color_path.astype(np.float32)
        color /= 255.0
    else:
        raise TypeError("color_path 既不是字符串路径也不是 NumPy 数组！")

    # 2. 加载 depth（可能是路径，也可能是数组）
    if isinstance(depth_path, str):
        depth_img = Image.open(depth_path)
        depth = np.array(depth_img)
    elif isinstance(depth_path, np.ndarray):
        depth = depth_path
    else:
        raise TypeError("depth_path 既不是字符串路径也不是 NumPy 数组！")

    # 3. 加载 mask（可能是路径，也可能是数组）
    if isinstance(mask_path, str):
        workspace_mask = np.array(Image.open(mask_path))
    elif isinstance(mask_path, np.ndarray):
        workspace_mask = mask_path
    else:
        raise TypeError("mask_path 既不是字符串路径也不是 NumPy 数组！")

    print("\n=== 尺寸验证 ===")
    print("深度图尺寸:", depth.shape[::-1])
    print("颜色图尺寸:", color.shape[:2][::-1])
    print("工作空间尺寸:", workspace_mask.shape[::-1])
    print("相机参数预设尺寸:", (640, 480))

    camera = CameraInfo(
        width=640,
        height=480,
        fx=DEPTH_INTR['fx'],
        fy=DEPTH_INTR['fy'],
        cx=DEPTH_INTR['ppx'],
        cy=DEPTH_INTR['ppy'],
        scale=DEPTH_FACTOR
    )

    print(f"workspace_mask shape: {workspace_mask.shape}, dtype: {workspace_mask.dtype}, unique: {np.unique(workspace_mask)}")
    print(f"depth shape: {depth.shape}, dtype: {depth.dtype}, min: {depth.min()}, max: {depth.max()}")

    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    mask = (workspace_mask > 0) & (depth > 0)
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    if len(cloud_masked) >= NUM_POINT:
        idxs = np.random.choice(len(cloud_masked), NUM_POINT, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), NUM_POINT - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]

    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device)
    end_points = {'point_clouds': cloud_sampled}

    return end_points, cloud_o3d

# ==================== 碰撞检测 ====================
def collision_detection(gg, cloud_points):
    mfcdetector = ModelFreeCollisionDetector(cloud_points, voxel_size=VOXEL_SIZE)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=COLLISION_THRESH)
    return gg[~collision_mask]

# ==================== 抓取位姿打印（可选） ====================
def print_grasp_poses(gg):
    print(f"\nTotal grasps after collision detection: {len(gg)}")
    for i, grasp in enumerate(gg):
        print(f"\nGrasp {i + 1}:")
        print(f"Position (x,y,z): {grasp.translation}")
        print(f"Rotation Matrix:\n{grasp.rotation_matrix}")
        print(f"Score: {grasp.score:.4f}")
        print(f"Width: {grasp.width:.4f}")

# ==================== 主函数：获取抓取预测 ====================
def run_grasp_inference(color_path, depth_path, mask_path):
    """
    主推理流程：
    1. 加载网络
    2. 处理数据并生成输入
    3. 进行抓取预测解码
    4. 碰撞检测
    5. 对抓取预测进行垂直角度筛选（仅保留接近垂直±30°的抓取）
    6. 打印/可视化抓取
    7. 返回前 50 个抓取中得分最高的抓取
    """
    # 1. 加载网络
    net = get_net()

    # 2. 处理数据
    end_points, cloud_o3d = get_and_process_data(color_path, depth_path, mask_path)

    # 3. 前向推理
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)

    # 4. 构造 GraspGroup 对象（这里 gg 是列表或类似列表的对象）
    gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())

    # 5. 碰撞检测
    if COLLISION_THRESH > 0:
        gg = collision_detection(gg, np.asarray(cloud_o3d.points))

    # 6. NMS 去重 + 按照得分排序（降序）
    gg.nms().sort_by_score()

    # ===== 新增筛选部分：对抓取预测的接近方向进行垂直角度限制 =====
    # 将 gg 转换为普通列表
    all_grasps = list(gg)
    vertical = np.array([0, 0, 1])  # 期望抓取接近方向（垂直桌面）
    angle_threshold = np.deg2rad(30)  # 30度的弧度值
    filtered = []
    for grasp in all_grasps:
        # 抓取的接近方向取 grasp.rotation_matrix 的第三列
        approach_dir = grasp.rotation_matrix[:, 0]
        # 计算夹角：cos(angle)=dot(approach_dir, vertical)
        cos_angle = np.dot(approach_dir, vertical)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        if angle < angle_threshold:
            filtered.append(grasp)
    if len(filtered) == 0:
        print("\n[Warning] No grasp predictions within vertical angle threshold. Using all predictions.")
        filtered = all_grasps
    else:
        print(
            f"\n[DEBUG] Filtered {len(filtered)} grasps within ±30° of vertical out of {len(all_grasps)} total predictions.")

    # 对过滤后的抓取根据 score 排序（降序）
    filtered.sort(key=lambda g: g.score, reverse=True)

    # 取前50个抓取（如果少于50个，则全部使用）
    top_grasps = filtered[:1]

    # 可视化过滤后的抓取，手动转换为 Open3D 物体
    grippers = [g.to_open3d_geometry() for g in top_grasps]
    print(f"\nVisualizing top {len(top_grasps)} grasps after vertical filtering...")
    o3d.visualization.draw_geometries([cloud_o3d, *grippers])

    # 选择得分最高的抓取（filtered 列表已按得分降序排序）
    best_grasp = top_grasps[0]
    best_translation = best_grasp.translation
    best_rotation = best_grasp.rotation_matrix
    best_width = best_grasp.width

    return best_translation, best_rotation, best_width


# ==================== 如果希望直接在此脚本中测试，可保留 main ====================
if __name__ == '__main__':
    # 示例：使用文件路径
    color_img_path = '/home/ncut/graspnet-baseline/doc/example/color.png'
    depth_img_path = '/home/ncut/graspnet-baseline/doc/example/depth.png'
    mask_img_path = '/home/ncut/graspnet-baseline/doc/example/mask1.png'

    t, R_mat, w = run_grasp_inference(color_img_path, depth_img_path, mask_img_path)
    print("\n=== 最优抓取信息 (文件路径输入) ===\nTranslation:", t, "\nRotation:\n", R_mat, "\nWidth:", w)
