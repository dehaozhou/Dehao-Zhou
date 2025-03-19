import yaml
from libs.auxiliary import create_folder_with_date, get_ip, popup_message
import sys
import cv2
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
from robotic_arm_package.robotic_arm import *
from vertical_grab.convert_d import convert_new
from cv_process import segment_image
from grasp_process import run_grasp_inference

# 相机内参
color_intr = {"ppx": 331.054, "ppy": 240.211, "fx": 604.248, "fy": 604.376}
depth_intr = {"ppx": 319.304, "ppy": 236.915, "fx": 387.897, "fy": 387.897}

# 手眼标定外参
# rotation_matrix = [
#     [0.00881983, -0.99903671, -0.04298679],
#     [0.99993794, 0.00910406, -0.00642086],
#     [0.00680603, -0.04292749, 0.99905501]
# ]
# translation_vector = [0.09830079, -0.04021631, -0.01756948]

# 手眼标定外参 新：20250315
rotation_matrix = [
    [-0.04208296,-0.9991138,-0.00080197],  # 相机的 X 轴基本与末端Y轴对齐
    [0.99898261,-0.04206438,-0.01625857],  # 相机 Y 轴大致与末端坐标系的负 X 轴平行
    [0.01621043,-0.00148536,0.9998675 ]    ## 相机 Z 轴基本与末端坐标系的 Z 轴保持一致
]
translation_vector = [0.0971208,-0.03754267, -0.01756948 + 0.005]

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

    error_code, joints, current_pose, arm_err_ptr, sys_err_ptr = robot.Get_Current_Arm_State()
    print("\n[DEBUG] 当前末端位姿:", current_pose)

    base_pose = convert_new(
        translation,
        rotation_mat_3x3,
        current_pose,
        rotation_matrix,
        translation_vector
    )
    print("[DEBUG] 基坐标系抓取位姿:", base_pose)

    # 首次运行只计算不执行
    if first_run:
        print("[INFO] 首次运行模拟完成，准备正式执行")
        first_run = False
        return  # 直接返回不执行后续动作

    # 正式执行部分
    base_pose_np = np.array(base_pose, dtype=float)
    base_xyz = base_pose_np[:3]
    base_rxyz = base_pose_np[3:]

    # 坐标调整
    #base_rxyz[2] = -base_rxyz[2]
    # base_rxyz[1] = 0
    # base_rxyz[2] = 0




    # 预抓取计算
    pre_grasp_offset = 0.1
    pre_grasp_pose = np.array(base_pose, dtype=float).copy()
    #rotation_mat = R.from_euler('xyz', pre_grasp_pose[3:]).as_matrix()
    rotation_mat = R.from_euler('ZYX', pre_grasp_pose[3:][::-1]).as_matrix()
    z_axis = rotation_mat[:, 2]
    pre_grasp_pose[:3] -= z_axis * pre_grasp_offset


    # 运动控制
    grasp_pose = np.concatenate([base_xyz, base_rxyz]).tolist()
    print(f"[DEBUG] 调整后的抓取位姿: {grasp_pose}")

    init = [5, 15, 10, -75, 0, 0, 0]
    fang = [-20, 25, 0, -90, 0, 25, 0]

    try:
        print(f"预抓取位姿: {pre_grasp_pose.tolist()}")
        ret = robot.Movej_P_Cmd(pre_grasp_pose.tolist(), 5)
        if ret != 0: raise RuntimeError(f"预抓取失败，错误码: {ret}")

        print(f"实际抓取: {grasp_pose}")
        ret = robot.Movej_P_Cmd(grasp_pose, 5)
        if ret != 0: raise RuntimeError(f"抓取失败，错误码: {ret}")

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
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    try:
        profile = pipeline.start(config)
        color_sensor = profile.get_device().query_sensors()[1]
        color_sensor.set_option(rs.option.enable_auto_exposure, 1)
        align = rs.align(rs.stream.color)  # 对齐到彩色图像流
    
        while True:
            frames = pipeline.wait_for_frames()
            if not frames: continue
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame: continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            callback(color_image, depth_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break

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
    init = [5, 15, 10, -75, 0, 0, 0]
    robot.Movej_Cmd(init, 10, 0)

    # 重置首次运行标志
    first_run = True
    displayD435()


if __name__ == "__main__":
    main()
