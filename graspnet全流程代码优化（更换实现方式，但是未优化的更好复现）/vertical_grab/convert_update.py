import numpy as np
from scipy.spatial.transform import Rotation as R
from robotic_arm_package.robotic_arm import *


def convert_new(
        grasp_translation,  # GraspNet输出的平移向量（相机坐标系）
        grasp_rotation_mat,  # GraspNet输出的旋转矩阵（相机坐标系，3x3）
        current_pose,  # 机械臂末端当前姿态 [x, y, z, rx, ry, rz]（基座坐标系）
        handeye_rot,  # 手眼标定旋转矩阵（相机->末端）
        handeye_trans,  # 手眼标定平移向量（相机->末端）
        T_ee2base
):
    """
    优化后的坐标系转换函数，主要改动：
    1. 修正坐标系转换链路顺序
    2. 处理GraspNet夹爪坐标系定义差异
    返回 [base_x, base_y, base_z, base_rx, base_ry, base_rz]
    """
    # ================== 坐标系对齐预处理 ==================
    # 修正GraspNet坐标系定义（X轴朝向 -> Z轴朝向）
    R_adjust = np.array([
        [0, 0, 1],  # 将X轴旋转到Z轴
        [0, 1, 0],  # Y轴保持不动
        [-1, 0, 0]  # Z轴旋转到-X轴
    ], dtype=np.float32)

    # 调整后的抓取姿态（相机坐标系）
    adjusted_rotation = grasp_rotation_mat @   R_adjust
    adjusted_translation = grasp_translation 
    # ================== 坐标系转换链路 ==================
    # 1. 构造抓取位姿的齐次矩阵（相机坐标系）
    T_grasp2cam = np.eye(4)
    T_grasp2cam[:3, :3] = adjusted_rotation
    T_grasp2cam[:3, 3] = adjusted_translation

    # 2. 手眼标定矩阵（相机->末端）
    T_cam2ee = np.eye(4)
    T_cam2ee[:3, :3] = handeye_rot
    T_cam2ee[:3, 3] = handeye_trans

    # 3. 获取当前末端到基座的变换矩阵（注意矩阵求逆）

    #（1）数学方法
    # x, y, z, rx, ry, rz = current_pose
    # T_ee2base = np.eye(4)
    # T_ee2base[:3, :3] = R.from_euler('XYZ', [rx, ry, rz]).as_matrix() 
    # T_ee2base[:3, 3] = [x, y, z]

    # （2）调用官方api去计算

    # 修正欧拉角顺序为机械臂实际使用的顺序（示例为XYZ顺序）


    # 4. 计算完整的转换链路：T_grasp2base = T_ee2base @ T_cam2ee @ T_grasp2cam
    T_grasp2base = T_ee2base @ T_cam2ee @ T_grasp2cam

    # ================== 结果解析 ==================


    # 齐次矩阵转位姿  注释掉的是数学方法

    # 提取平移分量
    # base_position = T_grasp2base[:3, 3]
    # # 提取并修正旋转分量（根据机械臂实际需求调整欧拉角顺序）
    # base_rotation = R.from_matrix(T_grasp2base[:3, :3])
    # base_euler = base_rotation.as_euler('XYZ', degrees=False)  # 示例使用ZYX顺序

    return T_grasp2base