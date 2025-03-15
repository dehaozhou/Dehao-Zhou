import numpy as np
from scipy.spatial.transform import Rotation as R

def convert_new(
    grasp_translation,   # GraspNet输出的平移向量（相机坐标系）
    grasp_rotation_mat,  # GraspNet输出的旋转矩阵（相机坐标系，3x3）
    current_pose,        # 机械臂末端当前姿态 [x, y, z, rx, ry, rz]（基座坐标系）
    handeye_rot,         # 手眼标定旋转矩阵（相机->末端）
    handeye_trans,       # 手眼标定平移向量（相机->末端）
    gripper_length=0.1 # 夹爪长度，默认0.1米（10cm）
):
    """
    将 GraspNet 输出的抓取位姿从相机坐标系转换到机械臂基座坐标系。
    返回 [base_x, base_y, base_z, base_rx, base_ry, base_rz]，其中 base_rx,ry,rz 为XYZ顺序的欧拉角。
    动态补偿夹爪长度，使夹爪指尖而非末端法兰中心对准目标抓取点。
    """
    # 1) 构造相机坐标系到末端坐标系的变换（手眼标定结果）
    T_cam2ee = np.eye(4, dtype=float)
    T_cam2ee[:3, :3] = handeye_rot
    T_cam2ee[:3, 3]  = handeye_trans

    # 2) 构造基座坐标系到当前末端姿态的变换
    x1, y1, z1, rx1, ry1, rz1 = current_pose
    T_base2ee = np.eye(4, dtype=float)
    # 注意末端姿态的欧拉角使用XYZ顺序
    ee_rotmat = R.from_euler('ZYX', [rz1, ry1, rx1], degrees=False).as_matrix()
    T_base2ee[:3, :3] = ee_rotmat
    T_base2ee[:3, 3]  = [x1, y1, z1]

    # 3) 构造相机坐标系到抓取姿态的变换（GraspNet预测值）
    T_cam2grasp = np.eye(4, dtype=float)
    T_cam2grasp[:3, :3] = grasp_rotation_mat
    T_cam2grasp[:3, 3]  = grasp_translation

    # 4) 计算基座坐标系到抓取姿态的变换
    #    数学链路: T_base2grasp = T_base2ee * (T_cam2ee * T_cam2grasp)
    T_base2grasp = T_base2ee.dot(T_cam2ee.dot(T_cam2grasp))

    # 提取抓取位姿的平移和旋转部分
    base_x, base_y, base_z = T_base2grasp[:3, 3]
    base_rotmat = T_base2grasp[:3, :3]

    # 5) 计算夹爪长度补偿: 根据末端姿态，将夹爪长度沿抓取朝向进行偏移
    # 假设GraspNet输出的旋转矩阵中，夹爪**朝向目标**的轴为局部坐标的 X 轴正方向
    # 因此取局部坐标系中的偏移向量 [-gripper_length, 0, 0]（沿 -X 方向后退）
    offset_local = np.array([-gripper_length, 0, 0], dtype=float)
    # 将偏移向量从局部抓取坐标系转换到基座坐标系
    offset_global = base_rotmat.dot(offset_local)
    # 用该偏移修正基座下的位置，使末端后移
    base_x += offset_global[0]
    base_y += offset_global[1]
    base_z += offset_global[2]

    # 6) 将基座下的抓取旋转矩阵转换为机械臂末端工具坐标系的旋转表示
    # （根据机械臂坐标系与末端工具坐标系的固定偏差进行修正）
    R_base_to_ee = np.array([
                                [0, 0, 1],   # 基座X → 末端Z
                                [0, -1, 0],  # 基座Y → 末端-Y
                                [1, 0, 0]    # 基座Z → 末端X
                            ], dtype=float)
    base_rotmat_corrected = R_base_to_ee.dot(base_rotmat)
    base_euler = R.from_matrix(base_rotmat_corrected).as_euler('ZYX', degrees=False)


    # 输出 [x, y, z, rx, ry, rz]
    result = [base_x, base_y, base_z, base_euler[0], base_euler[1], base_euler[2]]
    return result


