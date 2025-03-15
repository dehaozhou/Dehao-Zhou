import yaml  # 导入 YAML 库，用于读取配置文件
from libs.auxiliary import create_folder_with_date, get_ip, popup_message  # 从辅助库中导入相关函数
import sys  # 导入 sys 模块，用于操作系统相关功能
from robotic_arm_package.robotic_arm import *  # 导入机械臂控制相关的函数和类
import cv2  # 导入 OpenCV 库，用于图像处理
from vertical_grab.interface import vertical_catch  # 导入垂直抓取相关函数
import numpy as np  # 导入 NumPy 库，用于数组操作
import pyrealsense2 as rs  # 导入 RealSense 库，用于深度相机操作
import sys  # 再次导入 sys，确保系统路径更新
from cv_process import segment_image


# 相机内参（用于图像处理中的深度和RGB图像转换）
color_intr = {"ppx": 331.054, "ppy": 240.211, "fx": 604.248, "fy": 604.376}  # RGB相机内参
depth_intr = {"ppx": 319.304 , "ppy": 236.915, "fx": 387.897, "fy": 387.897}  # 深度相机内参

arm_gripper_length = 0.8  # 定义机械臂抓手的长度

vertical_rx_ry_rz = [3.14, 0, 0]  # 定义垂直抓取时的旋转角度,欧拉角代表绕三个x,y,z轴进行旋转
                                                      # 绕z轴转180度

# 定义相机和物体间的旋转矩阵（转换坐标）
rotation_matrix = [[0.00881983, -0.99903671, -0.04298679],
                   [0.99993794,  0.00910406, -0.00642086],
                   [0.00680603, -0.04292749, 0.99905501]]  # 旋转矩阵

# 定义物体的平移向量
translation_vector = [0.09830079, -0.04021631, -0.01756948]

# 定义全局变量，用于存储相机捕获的图像
global color_img, depth_img
color_img = None  # RGB图像
depth_img = None  # 深度图像

# 回调函数，用于处理相机帧数据
def callback(color_frame, depth_frame):
    global color_img, depth_img  # 使用全局变量存储图像数据
    # 图像缩放因子
    scaling_factor_x = 1  # 默认不缩放
    scaling_factor_y = 1 


    # 将图像按缩放因子缩放
    color_img = cv2.resize(color_frame, None, fx=scaling_factor_x, fy=scaling_factor_y, interpolation=cv2.INTER_AREA)
    depth_img = cv2.resize(depth_frame, None, fx=scaling_factor_x, fy=scaling_factor_y, interpolation=cv2.INTER_AREA)

    # 显示RGB图像
    cv2.imshow("RGB Image", color_img)
    k = cv2.waitKey(30) & 0xFF  # 等待按键
    cv2.imshow("Depth Image", depth_img)  # 显示深度图像
    k = cv2.waitKey(30) & 0xFF  # 等待按键

    # 如果RGB图像和深度图像都不为空，调用垂直抓取函数
    if color_img is not None and depth_img is not None:
         test_vertical_catch()  # 测试垂直抓取

    # 等待按键退出
    k = cv2.waitKey(1) & 0xFF  # 按键判断
    if k == ord('q'):  # 如果按下 'q' 键退出
        cv2.destroyAllWindows()  # 关闭所有OpenCV窗口

# 垂直抓取测试函数
def test_vertical_catch():
    # 检查图像是否更新
    if color_img is None or depth_img is None:
        print("等待图像数据更新...")
        return  # 如果图像数据为空，返回等待
    init = [10, 15, 10, -75, 0, 0, 0]  # 初始位姿
    fang = [-20, 25, 0, -90, 0, 25, 0]  # 放置位姿

    # 处理检测结果，获得掩码等信息
    mask = segment_image(color_img)

    # 如果掩码存在，进行后续处理
    if mask is not None:
        mask = mask.astype(np.uint8)  # 确保掩码为二值图像（0和255）
        print("掩码尺寸:", mask.shape)  # 打印掩码的尺寸
        scaling_factor = 1  # 缩放因子
        mask = cv2.resize(mask, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

        # 将掩码应用到原图，仅保留掩码区域
        masked_img = cv2.bitwise_and(color_img, color_img, mask=mask)  # 黑色区域不变
        masked_img[mask == 255] = [255, 255, 255]  # 掩码区域替换为白色

        # 显示最终的掩码效果图
        cv2.imshow("Masked Image", masked_img)
        cv2.waitKey(0)

    else:
        print("掩码为空，无法进行操作")  # 如果没有掩码，输出提示信息

    # 获取当前机械臂位姿
    error_code, joints, current_pose, arm_err_ptr, sys_err_ptr = robot.Get_Current_Arm_State()
    print("current_pose")
    print(current_pose)  # 打印当前位姿信息


    # 计算出垂直抓取的三个关键节点
    above_object_pose, correct_angle_pose, finally_pose = vertical_catch(
        mask, depth_img, color_intr, current_pose, arm_gripper_length, vertical_rx_ry_rz,
        rotation_matrix, translation_vector, True  # 调用垂直抓取函数，获取抓取位姿
    )
    # 打印三个关键位姿
    print("位点1")
    print(above_object_pose)
    print("#位点2")
    print(correct_angle_pose)
    print("#位点3")
    print(finally_pose)
    robot.Movej_Cmd(init, 10, 0)
    # 定义抓取初始位姿和放置位姿


    # 控制机械臂进行抓取操作
    robot.Movej_P_Cmd(above_object_pose, 10)  # 移动到第一个抓取位点
    robot.Movej_P_Cmd(correct_angle_pose, 10)  # 移动到第二个角度调整位点
    robot.Set_Gripper_Pick(200, 25)  # 启动机械臂夹爪进行抓取
    robot.Movej_Cmd(init, 10, 0)  # 移动到初始位姿
    robot.Movej_Cmd(fang, 10, 0)  # 移动到放置位姿
    robot.Set_Gripper_Release(100)  # 释放夹爪
    robot.Movej_Cmd(init, 10, 0)  # 返回到初始位姿

# 显示 RealSense 相机画面
def displayD435():
    # 初始化 RealSense 相机
    pipeline = rs.pipeline()  # 启动相机流
    config = rs.config()  # 控制相机流

    # 启用RGB图像流和深度图像流
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    try:
        pipeline.start(config)  # 启动相机流
    
    except Exception as e:
        print(f"相机连接异常：{e}")
        sys.exit(1)  # 如果启动失败，输出错误并退出



    try:
        while True:
            # 获取当前帧数据
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()  # 获取RGB图像帧
            depth_frame = frames.get_depth_frame()  # 获取深度图像帧

            if not color_frame or not depth_frame:  # 如果图像数据为空，跳过
                continue

            # 将图像帧转为NumPy数组
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # 调用回调函数处理图像数据
            callback(color_image, depth_image)

    finally:
        pipeline.stop()  # 停止相机流
        cv2.destroyAllWindows()  # 关闭所有OpenCV窗口

# 主函数
if __name__ == "__main__":
    robot_ip = get_ip()  # 获取机器人IP地址
    logger_.info(f'robot_ip:{robot_ip}')  # 打印机器人IP地址

    if robot_ip:  # 如果成功获取到机器人IP
        with open("config.yaml", 'r', encoding='utf-8') as file:  # 打开配置文件
            data = yaml.safe_load(file)  # 读取配置文件内容

        ROBOT_TYPE = data.get("ROBOT_TYPE")  # 获取机器人类型

        robot = Arm(ROBOT_TYPE, robot_ip)  # 初始化机械臂对象

        robot.Change_Work_Frame()  # 改变工作坐标系

        # 打印API版本信息
        print(robot.API_Version())

    else:  # 如果未能获取到机器人IP
        popup_message("提醒", "机械臂ip没有ping通")  # 弹出提示框
        sys.exit(1)  # 退出程序
    
    displayD435()  # 启动相机并开始图像处理
