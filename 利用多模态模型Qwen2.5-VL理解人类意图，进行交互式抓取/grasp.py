import yaml
from libs.auxiliary import create_folder_with_date, get_ip, popup_message
import sys
import cv2
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
from robotic_arm_package.robotic_arm import *
from convert_d import convert_new
from vlm_process import segment_image
from grasp_process import run_grasp_inference

# 相机内参  640*480
color_intr = {"ppx": 331.054, "ppy": 240.211, "fx": 604.248, "fy": 604.376}
depth_intr = {"ppx": 319.304, "ppy": 236.915, "fx": 387.897, "fy": 387.897}


# 相机内参  1280*720
color_intr = {"ppx": 656.581, "ppy": 360.316, "fx": 906.373, "fy": 906.563}
depth_intr = {"ppx": 638.839, "ppy": 354.818, "fx": 646.495, "fy": 646.495}


# # 手眼标定外参
# rotation_matrix = [
#     [0.00881983, -0.99903671, -0.04298679],
#     [0.99993794, 0.00910406, -0.00642086],
#     [0.00680603, -0.04292749, 0.99905501]
# ]
# translation_vector = [0.09830079, -0.04021631, -0.01756948]

#手眼标定外参 新：20250316
# rotation_matrix = [
#     [-0.05578176, -0.99668141, -0.0592837],  # 相机的 X 轴基本与末端Y轴对齐
#     [0.99840848 ,-0.05518776 ,-0.01161139],  # 相机 Y 轴大致与末端坐标系的负 X 轴平行
#     [0.00830112 ,-0.05983705  ,0.99817364]    ## 相机 Z 轴基本与末端坐标系的 Z 轴保持一致
# ]
# translation_vector = [0.12020577,-0.0436675, -0.00557289]

rotation_matrix =   [[-0.02953301, -0.99954574,  0.0060091 ],
 [ 0.99956198, -0.0295439,  -0.00173118],
 [ 0.00190793 , 0.00595535,  0.99998045]]

translation_vector =   [ 0.09349536, -0.03336514,-0.00571604]


# 第一个位置，会使得如果 + 会沿x轴正向移动

# 第二个位置，会使得如果 - 会沿y轴负向移动

# 全局变量
global color_img, depth_img, robot, first_run
color_img = None
depth_img = None
robot = None
first_run = True  # 新增首次运行标志

def get_aligned_frame(self):
        align = rs.align(rs.stream.color)  # type: ignore
        frames = self.pipline.wait_for_frames()
        # aligned_frames 对齐之后结果
        aligned_frames = align.process(frames)
        color = aligned_frames.get_color_frame()
        depth = aligned_frames.get_depth_frame()
        return color, depth

def callback(color_frame, depth_frame):
    global color_img, depth_img
    scaling_factor_x = 1
    scaling_factor_y = 1

    color_img = cv2.resize(
        color_frame, None,
        fx=scaling_factor_x,
        fy=scaling_factor_y,
        interpolation=cv2.INTER_AREA
    )
    depth_img = cv2.resize(
        depth_frame, None,
        fx=scaling_factor_x,
        fy=scaling_factor_y,
        interpolation=cv2.INTER_NEAREST
    )

    if color_img is not None and depth_img is not None:
        test_grasp()

def pose_to_list(pose):
    # 从 Pose 对象中提取位置和欧拉角信息
    x = pose.position.x
    y = pose.position.y
    z = pose.position.z
    rx = pose.euler.rx
    ry = pose.euler.ry
    rz = pose.euler.rz
    return [x, y, z, rx, ry, rz]

def matrix_to_list(T1):
    # 从 Pose 对象中提取位置和欧拉角信息
    T_ee2base =  T1.data
    return T_ee2base

def numpy_to_Matrix(nparr):
    # 假设 nparr 是一个 4x4 的 numpy 数组
    mat = Matrix()
    mat.irow = 4
    mat.iline = 4
    # 填充 data 字段
    for i in range(4):
        for j in range(4):
            mat.data[i][j] = float(nparr[i, j])
    return mat


def test_grasp():
    global color_img, depth_img, robot, first_run

    if color_img is None or depth_img is None:
        print("[WARNING] Waiting for image data...")
        return

    # 图像处理部分
    masks = segment_image(color_img)  
    
    translation, rotation_mat_3x3, width = run_grasp_inference(
        color_img,
        depth_img,
        masks
    )

    print(f"[DEBUG] Grasp预测结果 - 平移: {translation}, 旋转矩阵:\n{rotation_mat_3x3}")

    error_code, joints, current_pose_old, arm_err_ptr, sys_err_ptr = robot.Get_Current_Arm_State()
    print("\n[DEBUG]当前关节角度:", joints)
    print("\n[DEBUG]未补偿夹爪前位姿:", current_pose_old)

    # current_pose_first = robot.Algo_Cartesian_Tool(joints,0,0,-0.06)
    # current_pose = pose_to_list(current_pose_first)
    # print("[DEBUG] 补偿夹爪后的位姿:", current_pose)

    current_pose = current_pose_old

    T1 = robot.Algo_Pos2Matrix(current_pose)  # 位姿转换为齐次矩阵
    T_ee2base = matrix_to_list(T1)
    # print("[DEBUG] 官方api计算出对应的齐次矩阵:", T_ee2base)

   # 采取数学逻辑进行计算
    # base_pose = convert_new(
    #     translation,
    #     rotation_mat_3x3,
    #     current_pose,
    #     rotation_matrix,
    #     translation_vector,
    #     T_ee2base
    # )
    # print("[DEBUG] 基坐标系抓取位姿:", base_pose)

    T_grasp2base = convert_new(
        translation,
        rotation_mat_3x3,
        current_pose,
        rotation_matrix,
        translation_vector,
        T_ee2base
    )
    print("[DEBUG] 基坐标系抓取齐次矩阵:", T_grasp2base)

    matrix_struct = numpy_to_Matrix(T_grasp2base)

    base_pose_first = robot.Algo_Matrix2Pos(matrix_struct)
    # print("[DEBUG] base_pose_first是什么:", base_pose_first)

    base_pose = pose_to_list(base_pose_first)
    print("[DEBUG] 最终抓取位姿是什么:", base_pose)

    # 首次运行只计算不执行
    if first_run:
        print("[INFO] 首次运行模拟完成，准备正式执行")
        first_run = False
        return  # 直接返回不执行后续动作

    # 正式执行部分
    base_pose_np = np.array(base_pose, dtype=float)
    base_xyz = base_pose_np[:3]
    base_rxyz = base_pose_np[3:]

    # base_xyz[2] +=0.1
    # 坐标调整
    # base_rxyz[0] = 3.14
    # base_rxyz[1] = 0
    # base_rxyz[2] = 0




    # 实际抓取
    pre_grasp_offset = 0.08
    pre_grasp_pose = np.array(base_pose, dtype=float).copy()
    rotation_mat = R.from_euler('xyz', pre_grasp_pose[3:]).as_matrix()
    z_axis = rotation_mat[:, 2]
    pre_grasp_pose[:3] -= z_axis * pre_grasp_offset

    # 预抓取
    pre_grasp_offset_new = 0.25
    pre_grasp_pose_new = np.array(base_pose, dtype=float).copy()
    rotation_mat_new = R.from_euler('xyz', pre_grasp_pose_new[3:]).as_matrix()
    z_axis_new = rotation_mat_new[:, 2]
    pre_grasp_pose_new[:3] -= z_axis_new * pre_grasp_offset_new

    # pre_grasp_offset = 0.1
    # pre_grasp_pose = np.array(base_pose, dtype=float).copy()
    # pre_grasp_pose[2] += pre_grasp_offset

    # 运动控制
    grasp_pose = np.concatenate([base_xyz, base_rxyz]).tolist()
    print(f"[DEBUG] 调整后的抓取位姿: {grasp_pose}")

    #init = [5, 15, 10, -75, 0, 0, 0]s
    init = [0, -15, 0, -90, 0, -10, 0]

    fang = [-10, 18, -38, -89, 8.3, 15, 0]

    try:
        print(f"预抓取位姿: {pre_grasp_pose_new.tolist()}")
        ret = robot.Movej_P_Cmd(pre_grasp_pose.tolist(), 5)
        if ret != 0: raise RuntimeError(f"预抓取失败，错误码: {ret}")
        
        print(f"实际抓取: {pre_grasp_pose.tolist()}")
        ret = robot.Movej_P_Cmd(pre_grasp_pose.tolist(), 5)
        if ret != 0: raise RuntimeError(f"预抓取失败，错误码: {ret}")


        # print(f"实际抓取: {base_pose}")
        # ret = robot.Movej_P_Cmd(base_pose, 5)
        # if ret != 0: raise RuntimeError(f"抓取失败，错误码: {ret}")

        print("闭合夹爪")
        ret = robot.Set_Gripper_Pick(200, 300)
        if ret != 0: raise RuntimeError(f"夹爪闭合失败，错误码: {ret}")

        robot.Movej_Cmd(init, 10, 0)
        robot.Movej_Cmd(fang, 10, 0)
        robot.Set_Gripper_Release(200)
        robot.Movej_Cmd(init, 10, 0)
    except Exception as e:
        print(f"[ERROR] 运动异常: {str(e)}")
        robot.Movej_Cmd(init, 10, 0)



def displayD435():
    global first_run
    pipeline = rs.pipeline()
    config = rs.config()
    time.sleep(3)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    try:
        profile = pipeline.start(config)
        color_sensor = profile.get_device().query_sensors()[1]
        color_sensor.set_option(rs.option.enable_auto_exposure, 1)

        # 新增：创建对齐对象，将深度图与彩色图对齐
        align = rs.align(rs.stream.color)  # 对齐到彩色图像流

        while True:
            frames = pipeline.wait_for_frames()
            if not frames:
                continue

            # 对齐帧
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            callback(color_image, depth_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def main():
    global robot, first_run
    robot_ip = get_ip()
    logger_.info(f'robot_ip:{robot_ip}')

    if robot_ip:
        with open("config.yaml", 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        ROBOT_TYPE = data.get("ROBOT_TYPE")
        robot = Arm(ROBOT_TYPE, robot_ip)
        robot.Change_Work_Frame()
        print(robot.API_Version())
    else:
        popup_message("提醒", "机械臂 IP 没有 ping 通")
        sys.exit(1)

    # 初始化设置
    init = [0, -15, 0, -90, 0, -10, 0]
    robot.Movej_Cmd(init, 10, 0)

    # 重置首次运行标志
    first_run = True
    displayD435()


if __name__ == "__main__":
    def get_aligned_frame(self):
        align = rs.align(rs.stream.color)  # type: ignore
        frames = self.pipline.wait_for_frames()
        # aligned_frames 对齐之后结果
        aligned_frames = align.process(frames)
        color = aligned_frames.get_color_frame()
        depth = aligned_frames.get_depth_frame()
        return color, depth
    main()


    