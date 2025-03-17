import numpy as np
from scipy.spatial.transform import Rotation as R


def convert_new(
        grasp_translation,  # GraspNet 输出的平移 (相机坐标系下)
        grasp_rotation_mat,  # GraspNet 输出的旋转矩阵 (相机坐标系下, 3x3)
        current_ee_pose,  # 机械臂当前末端在基座坐标系下的位姿 [x, y, z, rx, ry, rz]
        handeye_rot,  # 手眼标定旋转矩阵 (相机→末端)
        handeye_trans,  # 手眼标定平移向量 (相机→末端)
        gripper_length=0.1
):
    """
    根据 GraspNet 输出 (相机系下的抓取位姿)，计算在机械臂基座系下的抓取位姿。

    -----------------------------------------------------------------------------------
    * GraspNet 默认抓取朝向是局部 x 轴，我们需要额外将其「旋转对齐」到机械臂末端 z 轴为抓取朝向。
    * 同时把夹爪长度的补偿 (往后退 gripper_length) 也放进该旋转对齐矩阵里，一并完成。
    * current_ee_pose 给出的末端姿态是在基座坐标系下 (x,y,z,rx,ry,rz)，
      若你的机械臂控制器/SDK 使用的是其它顺序，需要在此函数内对应修改 R.from_euler() 的顺序。
    -----------------------------------------------------------------------------------

    返回 [base_x, base_y, base_z, base_rx, base_ry, base_rz]，这里的 base_rx, base_ry, base_rz
    是按某种欧拉角顺序 (示例中是 'XYZ' 或 'ZYX') 输出，你可根据机械臂需求进行调整。
    """

    # =============== 1) 构造：GraspNet输出【抓取坐标系 → 相机坐标系】的变换矩阵 ================
    T_grasp2cam = np.eye(4, dtype=float)
    T_grasp2cam[:3, :3] = grasp_rotation_mat
    T_grasp2cam[:3, 3] = grasp_translation

    # =============== 2) 在 GraspNet 的输出上做「轴对齐 + 夹爪补偿」 ================
    #
    #   GraspNet 抓取坐标系：   x 轴 = 抓取主轴(夹爪张合方向),  y 轴 = 宽度方向,  z 轴 = 垂直于夹爪平面
    #   常见的机械臂抓取坐标系：z 轴 = 抓取主轴(夹爪张合方向).
    #
    #   所以需要一个 旋转矩阵 R_align：让 "GraspNet.x" 对齐到 "Robot.z"
    #   并且把 "GraspNet" 输出的抓取点再往后退 gripper_length (在新坐标系的 -Z).
    #
    #   如果把这个对齐+补偿独立做成 4x4 的矩阵 T_align ，那么
    #       T_gripper2cam = T_grasp2cam * T_align
    #   才是「已经对齐、包含补偿后的抓取姿态」在相机坐标系下的变换。
    #
    #   注：关于右乘还是左乘，取决于你要把此旋转当做“新的局部坐标系”怎样嵌入到原坐标系中。
    #       这里采用右乘的方式，让 T_align 表示 “新gripper坐标系 → 旧grasp坐标系” 的变换。
    #
    R_align = np.array([
        [0, 0, 1],  # newX = oldZ
        [0, 1, 0],  # newY = oldY
        [-1, 0, 0],  # newZ = -oldX  (或者相当于 oldX = -newZ)
    ], dtype=float)

    T_align = np.eye(4, dtype=float)
    T_align[:3, :3] = R_align
    #
    # gripper_length 补偿：假设新坐标系的 Z 是抓取主轴，因此让末端后退 (沿 -Z 方向)
    # 若想“探出去”，可以改成 [0, 0, +gripper_length]
    #
    T_align[:3, 3] = [0, 0, -gripper_length]

    # 得到【修正后的】抓取姿态 (相机坐标系下)
    T_gripper2cam = T_grasp2cam @ T_align

    # =============== 3) 手眼标定：构造【相机坐标系 → 末端坐标系】 ================
    #   如果实际标定结果是 (末端→相机)，就需要再取逆；此处假设 handeye_rot, handeye_trans
    #   的确代表 “相机→末端”。
    #
    T_cam2ee = np.eye(4, dtype=float)
    T_cam2ee[:3, :3] = handeye_rot
    T_cam2ee[:3, 3] = handeye_trans

    # =============== 4) 当前末端姿态：构造【末端坐标系 → 基座坐标系】的变换 ================
    #   如果你的机械臂 API 返回的 (x,y,z,rx,ry,rz) 本身就表示“末端在基座系的位姿”，
    #   那么做法是：T_ee2base * [0,0,0,1] = [x,y,z,1]，并把欧拉角对应旋转填进去。
    #
    #   注意这里欧拉角顺序需要和你的机械臂或自己定义保持一致。
    #
    x_ee, y_ee, z_ee, rx_ee, ry_ee, rz_ee = current_ee_pose

    # 例：机械臂有些驱动器/SDK喜欢 'ZYX' 顺序，也有喜欢 'XYZ'。下面仅做示例：
    # 如果你确定 rx_ee, ry_ee, rz_ee 是以 "XYZ" 顺序，则要用 R.from_euler('XYZ',[...])。
    # 如果你确定是 "ZYX" ，则要 R.from_euler('ZYX',[rz_ee, ry_ee, rx_ee])。
    #
    # 以下演示用 'XYZ'，根据你实际情况来：
    R_ee2base = R.from_euler('XYZ', [rx_ee, ry_ee, rz_ee], degrees=False).as_matrix()

    T_ee2base = np.eye(4, dtype=float)
    T_ee2base[:3, :3] = R_ee2base
    T_ee2base[:3, 3] = [x_ee, y_ee, z_ee]

    # =============== 5) 计算最终【抓取坐标系(对齐后) → 基座坐标系】 ================
    #
    #   T_gripper2base = T_ee2base * (T_cam2ee * T_gripper2cam)
    #
    #   这样就把"修正后"的抓取位姿从相机系一路转换到基座系。
    #
    T_gripper2base = T_ee2base @ (T_cam2ee @ T_gripper2cam)

    # 分离出旋转 + 平移
    final_rot_mat = T_gripper2base[:3, :3]
    final_trans = T_gripper2base[:3, 3]

    # =============== 6) 将最终旋转矩阵变为欧拉角 (如果你需要欧拉角作为机械臂指令) ================
    #   再次强调，具体要什么顺序，需要和你的机械臂驱动匹配。
    #   演示这里输出 "XYZ" 顺序的 [rx, ry, rz]。
    #
    final_euler = R.from_matrix(final_rot_mat).as_euler('XYZ', degrees=False)
    base_rx, base_ry, base_rz = final_euler

    # 拼装输出 [x, y, z, rx, ry, rz]
    result = [
        final_trans[0],
        final_trans[1],
        final_trans[2],
        base_rx,
        base_ry,
        base_rz
    ]

    return result
