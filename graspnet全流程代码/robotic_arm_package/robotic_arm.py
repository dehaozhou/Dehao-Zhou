#!/usr/bin/env python
# -*- coding: utf-8 -*-
# cython: language_level=3

import ctypes
import logging
import sys
import os.path
import time
from enum import IntEnum
import platform

# 此处为了兼容绝对路径和相对路径写了多种导入方式，推荐用户根据包的结构选择一种清晰的导入方式
if __package__ is None or __package__ == '':
    # 当作为脚本运行时，__package__ 为 None 或者空字符串
    from log_setting import CommonLog
else:
    # 当作为模块导入时，__package__ 为模块的包名
    from .log_setting import CommonLog


# 库路径
module_path = os.path.abspath(__file__)
package_dir = os.path.dirname(module_path)
dllPath = os.path.join(package_dir, 'libs')

# End loader
if platform.machine() == "x86_64":
    dllPath = os.path.join(dllPath, 'linux_x86', 'libRM_Base.so.1.0.0')
elif sys.platform == "win32":
    if sys.maxsize > 2**32:
        dllPath = os.path.join(dllPath, 'win_64', 'RM_Base.dll')
    else:
        dllPath = os.path.join(dllPath, 'win_32', 'RM_Base.dll')
        print(dllPath)
else:
    dllPath = os.path.join(dllPath, 'linux_arm', 'libRM_Base.so.1.0.0')

logger_ = logging.getLogger(__name__)
logger_ = CommonLog(logger_)

# 定义机械臂型号
RM65 = 65
RML63_I = 631
RML63_II = 632
ECO65 = 651
RM75 = 75
ECO62 = 62
GEN72 = 72

ARM_DOF = 7
MOVEJ_CANFD_CB = 0x0001  # 角度透传非阻
MOVEP_CANFD_CB = 0x0002  # 位姿透传非阻
FORCE_POSITION_MOVE_CB = 0x0003  # 力位混合透传

errro_message = {1: '1: CONTROLLER_DATA_RETURN_FALSE', 2: "2: INIT_MODE_ERR", 3: '3: INIT_TIME_ERR',
                 4: '4: INIT_SOCKET_ERR', 5: '5: SOCKET_CONNECT_ERR', 6: '6: SOCKET_SEND_ERR', 7: '7: SOCKET_TIME_OUT',
                 8: '8: UNKNOWN_ERR', 9: '9: CONTROLLER_DATA_LOSE_ERR', 10: '10: CONTROLLER_DATE_ARR_NUM_ERR',
                 11: '11: WRONG_DATA_TYPE', 12: '12: MODEL_TYPE_ERR', 13: '13: CALLBACK_NOT_FIND',
                 14: '14: ARM_ABNORMAL_STOP',
                 15: '15: TRAJECTORY_FILE_LENGTH_ERR', 16: '16: TRAJECTORY_FILE_CHECK_ERR',
                 17: '17: TRAJECTORY_FILE_READ_ERR', 18: '18: CONTROLLER_BUSY', 19: '19: ILLEGAL_INPUT',
                 20: '20: QUEUE_LENGTH_FULL',
                 21: '21 CALCULATION_FAILED', 22: '22: FILE_OPEN_ERR', 23: '23: FORCE_AUTO_STOP',
                 24: '24: DRAG_TEACH_FLAG_FALSE', 25: '25: LISTENER_RUNNING_ERR'}


class POS_TEACH_MODES(IntEnum):
    X_Dir = 0  # X轴方向
    Y_Dir = 1  # Y轴方向
    Z_Dir = 2  # Z轴方向


class ARM_CTRL_MODES(IntEnum):
    None_Mode = 0,  # 无规划
    Joint_Mode = 1,  # 关节空间规划
    Line_Mode = 2,  # 笛卡尔空间直线规划
    Circle_Mode = 3,  # 笛卡尔空间圆弧规划
    Replay_Mode = 4,  # 拖动示教轨迹复现
    Moves_Mode = 5      # 样条曲线运动


class RobotType(IntEnum):
    RM65 = 0
    RM75 = 1
    RML63I = 2
    RML63II = 3
    RML63III = 4
    NANO = 5
    ECO65 = 6
    ECO62 = 7
    GEN72 = 8


class SensorType(IntEnum):
    B = 0
    ZF = 1
    SF = 2


class JOINT_STATE(ctypes.Structure):
    _fields_ = [
        # ("joint", ctypes.c_float * ARM_DOF),
        ("temperature", ctypes.c_float * ARM_DOF),
        ("voltage", ctypes.c_float * ARM_DOF),
        ("current", ctypes.c_float * ARM_DOF),
        ("en_state", ctypes.c_byte * ARM_DOF),
        ("err_flag", ctypes.c_uint16 * ARM_DOF),
        ("sys_err", ctypes.c_uint16),
    ]


class Quat(ctypes.Structure):
    _fields_ = [
        ('w', ctypes.c_float),
        ('x', ctypes.c_float),
        ('y', ctypes.c_float),
        ('z', ctypes.c_float)
    ]


class Pos(ctypes.Structure):
    _fields_ = [
        ('x', ctypes.c_float),
        ('y', ctypes.c_float),
        ('z', ctypes.c_float)
    ]


class Euler(ctypes.Structure):
    _fields_ = [
        ('rx', ctypes.c_float),
        ('ry', ctypes.c_float),
        ('rz', ctypes.c_float)
    ]


class Pose(ctypes.Structure):
    _fields_ = [
        ('position', Pos),  # 位置
        ('quaternion', Quat),  # 四元数
        ('euler', Euler)  # 欧拉角
    ]


class Matrix(ctypes.Structure):
    _fields_ = [
        ('irow', ctypes.c_short),
        ('iline', ctypes.c_short),
        ('data', (ctypes.c_float * 4) * 4)
    ]


class FRAME_NAME(ctypes.Structure):
    _fields_ = [('name', ctypes.c_char * 12)]


class FRAME(ctypes.Structure):
    _fields_ = [('frame_name', FRAME_NAME),  # 坐标系名称
                ('pose', Pose),  # 坐标系位姿
                ('payload', ctypes.c_float),  # 坐标系末端负载重量
                ('x', ctypes.c_float),  # 坐标系末端负载位置
                ('y', ctypes.c_float),  # 坐标系末端负载位置
                ('z', ctypes.c_float)]  # 坐标系末端负载位置


class POSE_QUAT(ctypes.Structure):
    _fields_ = [('px', ctypes.c_float),
                ('py', ctypes.c_float),
                ('pz', ctypes.c_float),
                ('w', ctypes.c_float),
                ('x', ctypes.c_float),
                ('y', ctypes.c_float),
                ('z', ctypes.c_float)]


class ExpandConfig(ctypes.Structure):
    _fields_ = [("rpm_max", ctypes.c_int),
                ("rpm_acc", ctypes.c_int),
                ("conversin_coe", ctypes.c_int),
                ("limit_min", ctypes.c_int),
                ("limit_max", ctypes.c_int)]


class WiFi_Info(ctypes.Structure):
    _fields_ = [("channel", ctypes.c_int),
                ("ip", ctypes.c_char * 16),
                ("mac", ctypes.c_char * 18),
                ("mask", ctypes.c_char * 16),
                ("mode", ctypes.c_char * 5),
                ("password", ctypes.c_char * 16),
                ("ssid", ctypes.c_char * 32)]


class CallbackData(ctypes.Structure):
    _fields_ = [
        ("sockhand", ctypes.c_int),  # 返回调用时句柄
        ("codeKey", ctypes.c_int),  # 调用透传接口类型
        ("errCode", ctypes.c_int),  # API解析错误码
        ("pose", Pose),  # 当前位姿
        ("joint", ctypes.c_float * 7),  # 当前关节角度
        ("nforce", ctypes.c_int),  # 力控方向上所受的力
        ("sys_err", ctypes.c_uint16)  # 系统错误
    ]


# Define the JointStatus structure
class JointStatus(ctypes.Structure):
    _fields_ = [
        ("joint_current", ctypes.c_float * ARM_DOF),
        ("joint_en_flag", ctypes.c_ubyte * ARM_DOF),
        ("joint_err_code", ctypes.c_uint16 * ARM_DOF),
        ("joint_position", ctypes.c_float * ARM_DOF),
        ("joint_temperature", ctypes.c_float * ARM_DOF),
        ("joint_voltage", ctypes.c_float * ARM_DOF)
    ]


# Define the ForceData structure
class ForceData(ctypes.Structure):
    _fields_ = [
        ("force", ctypes.c_float * 6),
        ("zero_force", ctypes.c_float * 6),
        ("coordinate", ctypes.c_int)
    ]


# Define the RobotStatus structure
class RobotStatus(ctypes.Structure):
    _fields_ = [
        ("errCode", ctypes.c_int),  # API解析错误码
        ("arm_ip", ctypes.c_char_p),  # 返回消息的机械臂IP
        ("arm_err", ctypes.c_uint16),  # 机械臂错误码
        ("joint_status", JointStatus),  # 当前关节状态
        ("force_sensor", ForceData),  # 力数据
        ("sys_err", ctypes.c_uint16),  # 系统错误吗
        ("waypoint", Pose)  # 路点信息
    ]


CANFD_Callback = ctypes.CFUNCTYPE(None, CallbackData)
RealtimePush_Callback = ctypes.CFUNCTYPE(None, RobotStatus)


class TrajectoryData(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_int),
        ("size", ctypes.c_int),
        ("speed", ctypes.c_int),
        ("trajectory_name", ctypes.c_char * 32)
    ]


class ProgramTrajectoryData(ctypes.Structure):
    _fields_ = [
        ("page_num", ctypes.c_int),
        ("page_size", ctypes.c_int),
        ("total_size", ctypes.c_int),
        ("vague_search", ctypes.c_char * 32),
        ("list", TrajectoryData * 100)
    ]


class ProgramRunState(ctypes.Structure):
    _fields_ = [
        ("run_state", ctypes.c_int),
        ("id", ctypes.c_int),
        ("plan_num", ctypes.c_int),
        ("loop_num", ctypes.c_int * 10),
        ("loop_cont", ctypes.c_int * 10),
        ("step_mode", ctypes.c_int),
        ("plan_speed", ctypes.c_int)
    ]


# 电子围栏名称
class ElectronicFenceNames(ctypes.Structure):
    _fields_ = [('name', ctypes.c_char * 12)]


# 电子围栏配置参数
class ElectronicFenceConfig(ctypes.Structure):
    _fields_ = [
        ("form", ctypes.c_int),  # 形状，1 表示立方体，2 表示点面矢量平面，3 表示球体
        ("name", ctypes.c_char * 12),  # 几何模型名称，不超过10个字节，支持字母、数字、下划线
        # 立方体
        ("x_min_limit", ctypes.c_float),  # 立方体基于世界坐标系 X 方向最小位置，单位 0.001m
        ("x_max_limit", ctypes.c_float),  # 立方体基于世界坐标系 X 方向最大位置，单位 0.001m
        ("y_min_limit", ctypes.c_float),  # 立方体基于世界坐标系 Y 方向最小位置，单位 0.001m
        ("y_max_limit", ctypes.c_float),  # 立方体基于世界坐标系 Y 方向最大位置，单位 0.001m
        ("z_min_limit", ctypes.c_float),  # 立方体基于世界坐标系 Z 方向最小位置，单位 0.001m
        ("z_max_limit", ctypes.c_float),  # 立方体基于世界坐标系 Z 方向最大位置，单位 0.001m
        # 点面矢量平面
        ("x1", ctypes.c_float),  # 表示点面矢量平面三点法中的第一个点坐标，单位 0.001m
        ("z1", ctypes.c_float),
        ("y1", ctypes.c_float),
        ("x2", ctypes.c_float),  # 表示点面矢量平面三点法中的第二个点坐标，单位 0.001m
        ("y2", ctypes.c_float),
        ("z2", ctypes.c_float),
        ("x3", ctypes.c_float),  # 表示点面矢量平面三点法中的第三个点坐标，单位 0.001m
        ("y3", ctypes.c_float),
        ("z3", ctypes.c_float),
        # 球体
        ("radius", ctypes.c_float),  # 表示半径，单位 0.001m
        ("x", ctypes.c_float),  # 表示球心在世界坐标系 X 轴、Y轴、Z轴的坐标，单位 0.001m
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
    ]

    def to_output(self):
        name = self.name.decode("utf-8").strip()  # 去除字符串两端的空白字符
        output_dict = {"name": name}

        if self.form == 1:  # 立方体
            output_dict.update({
                "form": "cube",
                "x_min_limit": float(format(self.x_min_limit, ".3f")),
                "x_max_limit": float(format(self.x_max_limit, ".3f")),
                "y_min_limit": float(format(self.y_min_limit, ".3f")),
                "y_max_limit": float(format(self.y_max_limit, ".3f")),
                "z_min_limit": float(format(self.z_min_limit, ".3f")),
                "z_max_limit": float(format(self.z_max_limit, ".3f")),
            })
        elif self.form == 2:  # 点面矢量平面
            output_dict.update({
                "form": "point_face_vector_plane",
                "x1": float(format(self.x1, ".3f")),
                "y1": float(format(self.y1, ".3f")),
                "z1": float(format(self.z1, ".3f")),
                "x2": float(format(self.x2, ".3f")),
                "y2": float(format(self.y2, ".3f")),
                "z2": float(format(self.z2, ".3f")),
                "x3": float(format(self.x3, ".3f")),
                "y3": float(format(self.y3, ".3f")),
                "z3": float(format(self.z3, ".3f")),
            })
        elif self.form == 3:  # 球体
            output_dict.update({
                "form": "sphere",
                "radius": float(format(self.radius, ".3f")),
                "x": float(format(self.x, ".3f")),
                "y": float(format(self.y, ".3f")),
                "z": float(format(self.z, ".3f")),
            })

        return output_dict


# 夹爪状态
class GripperState(ctypes.Structure):
    _fields_ = [
        ("enable_state", ctypes.c_bool),  # 夹爪使能标志，0 表示未使能，1 表示使能
        ("status", ctypes.c_int),  # 夹爪在线状态，0 表示离线， 1表示在线
        # 夹爪错误信息，低8位表示夹爪内部的错误信息bit5-7 保留bit4 内部通bit3 驱动器bit2 过流 bit1 过温bit0
        ("error", ctypes.c_int),
        # 当前工作状态：1 夹爪张开到最大且空闲，2 夹爪闭合到最小且空闲，3 夹爪停止且空闲，4 夹爪正在闭合，5 夹爪正在张开，6 夹爪
        ("mode", ctypes.c_int),
        ("current_force", ctypes.c_int),  # 夹爪当前的压力，单位g
        ("temperature", ctypes.c_int),  # 当前温度，单位℃
        ("actpos", ctypes.c_int),  # 夹爪开口度
    ]


class CtrlInfo(ctypes.Structure):
    _fields_ = [
        ("build_time", ctypes.c_char * 20),
        ("version", ctypes.c_char * 10),
    ]


class DynamicInfo(ctypes.Structure):
    _fields_ = [
        ("model_version", ctypes.c_char * 5),
    ]


class PlanInfo(ctypes.Structure):
    _fields_ = [
        ("build_time", ctypes.c_char * 20),
        ("version", ctypes.c_char * 10),
    ]


class AlgorithmInfo(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_char * 10),
    ]


# 机械臂软件信息
class ArmSoftwareInfo(ctypes.Structure):
    _fields_ = [
        ("product_version", ctypes.c_char * 10),
        ("algorithm_info", AlgorithmInfo),
        ("ctrl_info", CtrlInfo),
        ("dynamic_info", DynamicInfo),
        ("plan_info", PlanInfo),
    ]


# 定义ToolEnvelope结构体
class ToolEnvelope(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 12),
        ("radius", ctypes.c_float),  # 工具包络球体的半径，单位 m
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
    ]

    def __init__(self, name=None, radius=None, x=None, y=None, z=None):
        if all(param is None for param in [name, radius, x, y, z]):
            return
        else:
            # 转换name
            self.name = name.encode('utf-8')
            self.radius = radius
            self.x = x
            self.y = y
            self.z = z

    def to_output(self):
        name = self.name.decode("utf-8")
        # 创建一个字典，包含ToolEnvelope的所有属性
        output_dict = {
            "name": name,
            "radius": float(format(self.radius, ".3f")),
            "x": float(format(self.x, ".3f")),
            "y": float(format(self.y, ".3f")),
            "z": float(format(self.z, ".3f"))
        }
        return output_dict


# 定义ToolEnvelopeList结构体，其中包含一个ToolEnvelope数组
class ToolEnvelopeList(ctypes.Structure):
    _fields_ = [
        ("tool_name", ctypes.c_char * 12),  # 坐标系名称
        ("list", ToolEnvelope * 5),  # 包络参数列表，最多5个
        ("count", ctypes.c_int),  # 包络参数
    ]

    def __init__(self, tool_name=None, list=None, count=None):
        if all(param is None for param in [tool_name, list, count]):
            return
        else:
            # 转换tool_name
            self.tool_name = tool_name.encode('utf-8')

            self.list = (ToolEnvelope * 5)(*list)
            self.count = count

    def to_output(self):
        name = self.tool_name.decode("utf-8")

        output_dict = {
            "tool_name": name,
            "List": [self.list[i].to_output() for i in range(self.count)],
            "count": self.count,
        }
        return output_dict


class Waypoint(ctypes.Structure):
    _fields_ = [("point_name", ctypes.c_char * 12),
                ("joint", ctypes.c_float * ARM_DOF),
                ("pose", Pose),
                ("work_frame", ctypes.c_char * 12),
                ("tool_frame", ctypes.c_char * 12),
                ("time", ctypes.c_char * 20)]

    def __init__(self, point_name=None, joint=None, pose=None, work_frame=None, tool_frame=None, time=''):
        if all(param is None for param in [point_name, joint, pose, work_frame, tool_frame]):
            return
        else:
            # 转换point_name
            self.point_name = point_name.encode('utf-8')

            # 转换joint
            self.joint = (ctypes.c_float * ARM_DOF)(*joint)

            pose_value = Pose()
            pose_value.position = Pos(*pose[:3])
            pose_value.euler = Euler(*pose[3:])

            self.pose = pose_value

            # 转换work_frame和tool_frame
            self.work_frame = work_frame.encode('utf-8')
            self.tool_frame = tool_frame.encode('utf-8')

            # 转换time
            self.time = time.encode('utf-8')

    def to_output(self):
        name = self.point_name.decode("utf-8")
        wname = self.work_frame.decode("utf-8")
        tname = self.tool_frame.decode("utf-8")
        time = self.time.decode("utf-8")
        position = self.pose.position
        euler = self.pose.euler

        output_dict = {
            "point_name": name,
            "joint": [float(format(self.joint[i], ".3f")) for i in range(ARM_DOF)],
            "pose": [position.x, position.y, position.z, euler.rx, euler.ry, euler.rz],
            "work_frame": wname,
            "tool_frame": tname,
            "time": time,
        }
        return output_dict


# 定义WaypointsList结构体
class WaypointsList(ctypes.Structure):
    _fields_ = [("page_num", ctypes.c_int),
                ("page_size", ctypes.c_int),
                ("total_size", ctypes.c_int),
                ("vague_search", ctypes.c_char * 32),
                ("points_list", Waypoint * 100)]

    def to_output(self):
        vague_search = self.vague_search.decode("utf-8")
        non_empty_outputs = []
        for i in range(self.total_size):
            if self.points_list[i].point_name != b'':  # 判断列表是否为空
                output = self.points_list[i].to_output()
                non_empty_outputs.append(output)

        output_dict = {
            "total_size": self.total_size,
            "vague_search": vague_search,
            "points_list": non_empty_outputs,
        }
        return output_dict


class Set_Joint():
    def Set_Joint_Speed(self, joint_num, speed, block=True):
        """
         Set_Joint_Speed 设置关节最大速度
         ArmSocket socket句柄
         joint_num 关节序号，1~7
         speed 关节转速，单位：°/s
         block RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        return 0-成功，失败返回:错误码, rm_define.h查询.
        """
        self.pDll.Set_Joint_Speed.argtypes = (
            ctypes.c_int, ctypes.c_byte, ctypes.c_float, ctypes.c_bool)
        self.pDll.Set_Joint_Speed.restype = self.check_error

        tag = self.pDll.Set_Joint_Speed(self.nSocket, joint_num, speed, block)

        logger_.info(f'Set_Joint_Speed:{tag}')

        return tag

    def Set_Joint_Acc(self, joint_num, acc, block=True):
        """
        Set_Joint_Acc 设置关节最大加速度
        ArmSocket socket句柄
        joint_num 关节序号，1~7
        acc 关节转速，单位：°/s²
        block RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        return 0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Set_Joint_Acc.argtypes = (
            ctypes.c_int, ctypes.c_byte, ctypes.c_float, ctypes.c_bool)
        self.pDll.Set_Joint_Acc.restype = self.check_error

        tag = self.pDll.Set_Joint_Acc(self.nSocket, joint_num, acc, block)

        logger_.info(f'Set_Joint_Acc:{tag}')

        return tag

    def Set_Joint_Min_Pos(self, joint_num, joint, block=True):
        """
        Set_Joint_Min_Pos 设置关节最小限位
        ArmSocket socket句柄
        joint_num 关节序号，1~7
        joint 关节最小位置，单位：°
        block RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        return 0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Set_Joint_Min_Pos.argtypes = (
            ctypes.c_int, ctypes.c_byte, ctypes.c_float, ctypes.c_bool)
        self.pDll.Set_Joint_Min_Pos.restype = self.check_error

        tag = self.pDll.Set_Joint_Min_Pos(
            self.nSocket, joint_num, joint, block)

        logger_.info(f'Set_Joint_Min_Pos:{tag}')

        return tag

    def Set_Joint_Max_Pos(self, joint_num, joint, block=True):
        """
        Set_Joint_Max_Pos 设置关节最大限位
        ArmSocket socket句柄
        joint_num 关节序号，1~7
        joint 关节最小位置，单位：°
        block RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        return 0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Set_Joint_Max_Pos.argtypes = (
            ctypes.c_int, ctypes.c_byte, ctypes.c_float, ctypes.c_bool)
        self.pDll.Set_Joint_Max_Pos.restype = self.check_error

        tag = self.pDll.Set_Joint_Max_Pos(
            self.nSocket, joint_num, joint, block)

        logger_.info(f'Set_Joint_Max_Pos:{tag}')

        return tag

    def Set_Joint_Drive_Speed(self, joint_num, speed, block=True):
        """
         Set_Joint_Drive_Speed 设置关节最大速度(驱动器)
         ArmSocket socket句柄
         joint_num 关节序号，1~7
         speed 关节转速，单位：°/s
         block RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        return 0-成功，失败返回:错误码, rm_define.h查询.
        """
        self.pDll.Set_Joint_Drive_Speed.argtypes = (
            ctypes.c_int, ctypes.c_byte, ctypes.c_float, ctypes.c_bool)
        self.pDll.Set_Joint_Drive_Speed.restype = self.check_error

        tag = self.pDll.Set_Joint_Drive_Speed(
            self.nSocket, joint_num, speed, block)

        logger_.info(f'Set_Joint_Drive_Speed:{tag}')

        return tag

    def Set_Joint_Drive_Acc(self, joint_num, acc, block=True):
        """
        Set_Joint_Drive_Acc 设置关节最大加速度(驱动器)
        ArmSocket socket句柄
        joint_num 关节序号，1~7
        acc 关节转速，单位：°/s²
        block RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        return 0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Set_Joint_Drive_Acc.argtypes = (
            ctypes.c_int, ctypes.c_byte, ctypes.c_float, ctypes.c_bool)
        self.pDll.Set_Joint_Drive_Acc.restype = self.check_error

        tag = self.pDll.Set_Joint_Drive_Acc(
            self.nSocket, joint_num, acc, block)

        logger_.info(f'Set_Joint_Drive_Acc:{tag}')

        return tag

    def Set_Joint_Drive_Min_Pos(self, joint_num, joint, block=True):
        """
        Set_Joint_Drive_Min_Pos 设置关节最小限位(驱动器)
        ArmSocket socket句柄
        joint_num 关节序号，1~7
        joint 关节最小位置，单位：°
        block RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        return 0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Set_Joint_Drive_Min_Pos.argtypes = (
            ctypes.c_int, ctypes.c_byte, ctypes.c_float, ctypes.c_bool)
        self.pDll.Set_Joint_Drive_Min_Pos.restype = self.check_error

        tag = self.pDll.Set_Joint_Drive_Min_Pos(
            self.nSocket, joint_num, joint, block)

        logger_.info(f'Set_Joint_Drive_Min_Pos:{tag}')

        return tag

    def Set_Joint_Drive_Max_Pos(self, joint_num, joint, block=True):
        """
        Set_Joint_Drive_Max_Pos 设置关节最大限位(驱动器)
        ArmSocket socket句柄
        joint_num 关节序号，1~7
        joint 关节最小位置，单位：°
        block RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        return 0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Set_Joint_Drive_Max_Pos.argtypes = (
            ctypes.c_int, ctypes.c_byte, ctypes.c_float, ctypes.c_bool)
        self.pDll.Set_Joint_Drive_Max_Pos.restype = self.check_error

        tag = self.pDll.Set_Joint_Drive_Max_Pos(
            self.nSocket, joint_num, joint, block)

        logger_.info(f'Set_Joint_Drive_Max_Pos:{tag}')

        return tag

    def Set_Joint_EN_State(self, joint_num, state, block=True):
        """
        Set_Joint_EN_State 设置关节使能状态
        :param joint_num: 关节序号，1~7
        :param state: true-上使能，false-掉使能
        :param block:  RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:
        """

        self.pDll.Set_Joint_EN_State.astypes = (
            ctypes.c_int, ctypes.c_byte, ctypes.c_bool, ctypes.c_bool)
        self.pDll.restype = self.check_error

        tag = self.pDll.Set_Joint_EN_State(
            self.nSocket, joint_num, state, block)

        logger_.info(f'Set_Joint_EN_State:{tag}')

        return tag

    def Set_Joint_Zero_Pos(self, joint_num, block):
        """
        Set_Joint_Zero_Pos 将当前位置设置为关节零位
        :param joint_num: 关节序号，1~7
        :param block: RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Set_Joint_Zero_Pos.argtypes = (
            ctypes.c_int, ctypes.c_byte, ctypes.c_bool)
        self.pDll.Set_Joint_Zero_Pos.restype = self.check_error

        tag = self.pDll.Set_Joint_Zero_Pos(self.nSocket, joint_num, block)

        logger_.info(f'Set_Joint_Zero_Pos:{tag}')

        return tag

    def Set_Joint_Err_Clear(self, joint_num, block=True):
        """
        Set_Joint_Err_Clear 清楚关节错误
        :param joint_num: 关节序号，1~7
        :param block: RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Set_Joint_Err_Clear.argtypes = (
            ctypes.c_int, ctypes.c_byte, ctypes.c_bool)
        self.pDll.Set_Joint_Err_Clear.restype = self.check_error

        tag = self.pDll.Set_Joint_Err_Clear(self.nSocket, joint_num, block)

        logger_.info(f'Set_Joint_Err_Clear:{tag}')

        return tag

    def Auto_Set_Joint_Limit(self, limit_mode):
        """
        Auto_Set_Joint_Limit    自动设置关节限位
        :param limit_mode: 设置类型，1-正式模式，各关节限位为规格参数中的软限位和硬限位
        :param block: RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Auto_Set_Joint_Limit.argtypes = (ctypes.c_int, ctypes.c_byte)
        self.pDll.Auto_Set_Joint_Limit.restype = self.check_error

        tag = self.pDll.Auto_Set_Joint_Limit(self.nSocket, limit_mode)

        logger_.info(f'Auto_Set_Joint_Limit:{tag}')

        return tag

    def Auto_Fix_Joint_Over_Soft_Limit(self, block=True):
        """
        Auto_Fix_Joint_Over_Soft_Limit    超出限位后，自动运动到限位内
        :param block: RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """
        self.pDll.Auto_Fix_Joint_Over_Soft_Limit.argtypes = (
            ctypes.c_int, ctypes.c_bool)
        self.pDll.Auto_Fix_Joint_Over_Soft_Limit.restype = self.check_error

        tag = self.pDll.Auto_Fix_Joint_Over_Soft_Limit(self.nSocket, block)

        logger_.info(f'Auto_Fix_Joint_Over_Soft_Limit:{tag}')

        return tag


class Get_Joint():

    def Get_Joint_Speed(self, retry=0):
        """
        Get_Joint_Speed 查询关节最大速度
        :return:
        """
        le = self.code
        speed = (ctypes.c_float * le)()  # 关节1~7转速数组，单位：°/s
        tag = self.pDll.Get_Joint_Speed(self.nSocket, speed)

        while tag and retry:
            logger_.info(f'Get_Joint_Speed:{tag},retry is :{6 - retry}')
            tag = self.pDll.Get_Joint_Speed(self.nSocket, speed)
            retry -= 1

        logger_.info(f'Get_Joint_Speed:{tag}')

        return tag, list(speed)

    def Get_Joint_Acc(self, retry=0):
        """
        Get_Joint_Acc 查询关节最大加速度
        :return:
        """
        le = self.code
        acc = (ctypes.c_float * le)()  # 关节1~7加速度数组，单位：°/s²
        tag = self.pDll.Get_Joint_Acc(self.nSocket, acc)

        while tag and retry:
            logger_.info(f'Get_Joint_Acc:{tag},retry is :{6 - retry}')
            tag = self.pDll.Get_Joint_Acc(self.nSocket, acc)
            retry -= 1

        logger_.info(f'Get_Joint_Acc:{tag}')

        return tag, list(acc)

    def Get_Joint_Min_Pos(self, retry=0):
        """
        Get_Joint_Min_Pos 获取关节最小限位
        :return:
        """
        le = self.code
        min_joint = (ctypes.c_float * le)()  # 关节1~7最小位置数组，单位：°
        tag = self.pDll.Get_Joint_Min_Pos(self.nSocket, min_joint)

        while tag and retry:
            logger_.info(f'Get_Joint_Min_Pos:{tag},retry is :{6 - retry}')
            tag = self.pDll.Get_Joint_Min_Pos(self.nSocket, min_joint)
            retry -= 1

        logger_.info(f'Get_Joint_Min_Pos:{tag}')

        return tag, list(min_joint)

    def Get_Joint_Max_Pos(self, retry=0):
        """
        Get_Joint_Max_Pos 获取关节最大限位
        :return:

        """
        le = self.code
        max_joint = (ctypes.c_float * le)()  # 关节1~7最大位置数组，单位：°
        tag = self.pDll.Get_Joint_Max_Pos(self.nSocket, max_joint)

        while tag and retry:
            logger_.info(f'Get_Joint_Max_Pos:{tag},retry is :{6 - retry}')
            tag = self.pDll.Get_Joint_Max_Pos(self.nSocket, max_joint)
            retry -= 1

        logger_.info(f'Get_Joint_Max_Pos:{tag}')

        return tag, list(max_joint)

    def Get_Joint_Drive_Speed(self, retry=0):
        """
        Get_Joint_Drive_Speed 查询关节最大速度(驱动器)
        :return:
        """
        le = self.code
        speed = (ctypes.c_float * le)()  # 关节1~7转速数组，单位：°/s
        tag = self.pDll.Get_Joint_Drive_Speed(self.nSocket, speed)

        while tag and retry:
            logger_.info(f'Get_Joint_Drive_Speed:{tag},retry is :{6 - retry}')
            tag = self.pDll.Get_Joint_Drive_Speed(self.nSocket, speed)
            retry -= 1

        logger_.info(f'Get_Joint_Drive_Speed:{tag}')

        return tag, list(speed)

    def Get_Joint_Drive_Acc(self, retry=0):
        """
        Get_Joint_Drive_Acc 查询关节最大加速度(驱动器)
        :return:
        """
        le = self.code
        acc = (ctypes.c_float * le)()  # 关节1~7加速度数组，单位：°/s²
        tag = self.pDll.Get_Joint_Drive_Acc(self.nSocket, acc)

        while tag and retry:
            logger_.info(f'Get_Joint_Drive_Acc:{tag},retry is :{6 - retry}')
            tag = self.pDll.Get_Joint_Drive_Acc(self.nSocket, acc)
            retry -= 1

        logger_.info(f'Get_Joint_Drive_Acc:{tag}')

        return tag, list(acc)

    def Get_Joint_Drive_Min_Pos(self, retry=0):
        """
        Get_Joint_Drive_Min_Pos 获取关节最小限位(驱动器)
        :return:
        """
        le = self.code
        min_joint = (ctypes.c_float * le)()  # 关节1~7最小位置数组，单位：°
        tag = self.pDll.Get_Joint_Drive_Min_Pos(self.nSocket, min_joint)

        while tag and retry:
            logger_.info(
                f'Get_Joint_Drive_Min_Pos:{tag},retry is :{6 - retry}')
            tag = self.pDll.Get_Joint_Drive_Min_Pos(self.nSocket, min_joint)
            retry -= 1

        logger_.info(f'Get_Joint_Drive_Min_Pos:{tag}')

        return tag, list(min_joint)

    def Get_Joint_Drive_Max_Pos(self, retry=0):
        """
        Get_Joint_Drive_Max_Pos 获取关节最大限位(驱动器)
        :return:

        """
        le = self.code
        max_joint = (ctypes.c_float * le)()  # 关节1~7最大位置数组，单位：°
        tag = self.pDll.Get_Joint_Drive_Max_Pos(self.nSocket, max_joint)

        while tag and retry:
            logger_.info(
                f'Get_Joint_Drive_Max_Pos:{tag},retry is :{6 - retry}')
            tag = self.pDll.Get_Joint_Drive_Max_Pos(self.nSocket, max_joint)
            retry -= 1

        logger_.info(f'Get_Joint_Drive_Max_Pos:{tag}')

        return tag, list(max_joint)

    def Get_Joint_EN_State(self, retry=0):
        """
        Get_Joint_EN_State 获取关节使能状态
        :return:
        """
        le = self.code
        state = (ctypes.c_ubyte * le)()  # 关节1~7使能状态数组，1-使能状态，0-掉使能状态
        tag = self.pDll.Get_Joint_EN_State(self.nSocket, state)

        while retry:
            logger_.info(f'Get_Joint_EN_State:{tag},retry is :{6 - retry}')
            tag = self.pDll.Get_Joint_EN_State(self.nSocket, state)
            retry -= 1

        logger_.info(f'Get_Joint_EN_State:{tag}')
        return tag, list(state)

    def Get_Joint_Err_Flag(self, retry=0):
        """
        Get_Joint_Err_Flag 获取关节Err Flag
        :return:state  存放关节错误码（请参考api文档中的关节错误码）
        bstate   关节抱闸状态(1代表抱闸未打开，0代表抱闸已打开)
        """
        # le = int(str(self.code)[0])
        le = self.code

        state = (ctypes.c_uint16 * le)()
        bstate = (ctypes.c_uint16 * le)()

        tag = self.pDll.Get_Joint_Err_Flag(self.nSocket, state, bstate)

        while tag and retry:
            logger_.info(f'Get_Joint_Err_Flag:{tag},retry is :{6 - retry}')
            tag = self.pDll.Get_Joint_Err_Flag(self.nSocket, state, bstate)
            retry -= 1

        logger_.info(f'Get_Joint_Err_Flag:{tag}')
        return tag, list(state), list(bstate)

    def Get_Tool_Software_Version(self):
        """
        Get_Tool_Software_Version        查询末端接口板软件版本号
        :return:
        """
        version = ctypes.c_int()
        tag = self.pDll.Get_Tool_Software_Version(
            self.nSocket, ctypes.byref(version))

        logger_.info(f'Get_Tool_Software_Version:{tag}')
        return tag, hex(version.value)

    def Get_Joint_Software_Version(self):
        """
        Get_Joint_Software_Version       查询关节软件版本号
        :return:  关节软件版本号
        """

        if self.code == 6:
            self.pDll.Get_Joint_Software_Version.argtypes = (
                ctypes.c_int, ctypes.c_int * 6)
            self.pDll.Get_Joint_Software_Version.restype = self.check_error

            version = (ctypes.c_int * 6)()

        else:
            self.pDll.Get_Joint_Software_Version.argtypes = (
                ctypes.c_int, ctypes.c_int * 7)
            self.pDll.Get_Joint_Software_Version.restype = self.check_error

            version = (ctypes.c_int * 7)()

        tag = self.pDll.Get_Joint_Software_Version(self.nSocket, version)

        return tag, [hex(i) for i in version]


class Tcp_Config():
    def Set_Arm_Line_Speed(self, speed, block=True):
        """
        Set_Arm_Line_Speed 设置机械臂末端最大线速度
        :param speed: 末端最大线速度，单位m/s
        :param block: RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Set_Arm_Line_Speed.argtypes = (
            ctypes.c_int, ctypes.c_float, ctypes.c_bool)
        self.pDll.Set_Arm_Line_Speed.restype = self.check_error

        tag = self.pDll.Set_Arm_Line_Speed(self.nSocket, speed, block)

        logger_.info(f'Set_Arm_Line_Speed:{tag}')

        return tag

    def Set_Arm_Line_Acc(self, acc, block=True):
        """
        Set_Arm_Line_Acc 设置机械臂末端最大线加速度
        :param acc: 末端最大线加速度，单位m/s^2
        :param block: RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """
        self.pDll.Set_Arm_Line_Acc.argtypes = (
            ctypes.c_int, ctypes.c_float, ctypes.c_bool)
        self.pDll.Set_Arm_Line_Acc.restype = self.check_error

        tag = self.pDll.Set_Arm_Line_Acc(self.nSocket, acc, block)

        logger_.info(f'Set_Arm_Line_Acc: {tag}')

        return tag

    def Set_Arm_Angular_Speed(self, speed, block=True):
        """
        Set_Arm_Angular_Speed 设置机械臂末端最大角速度
        :param speed: 末端最大角速度，单位rad/s
        :param block: RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """
        self.pDll.Set_Arm_Angular_Speed.argtypes = (
            ctypes.c_int, ctypes.c_float, ctypes.c_bool)
        self.pDll.Set_Arm_Angular_Speed.restype = self.check_error

        tag = self.pDll.Set_Arm_Angular_Speed(self.nSocket, speed, block)

        logger_.info(f'Set_Arm_Angular_Speed: {tag}')

        return tag

    def Set_Arm_Angular_Acc(self, acc, block=True):
        """
        Set_Arm_Angular_Acc 设置机械臂末端最大角加速度
        :param acc: 末端最大角加速度，单位rad/s^2
        :param block: RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """
        self.pDll.Set_Arm_Angular_Acc.argtypes = (
            ctypes.c_int, ctypes.c_float, ctypes.c_bool)
        self.pDll.Set_Arm_Angular_Acc.restype = self.check_error

        tag = self.pDll.Set_Arm_Angular_Acc(self.nSocket, acc, block)

        logger_.info(f'Set_Arm_Angular_Acc: {tag}')

        return tag

    def Get_Arm_Line_Speed(self, retry=0):
        """
        Get_Arm_Line_Speed 获取机械臂末端最大线速度
        :return:
        """

        speed = ctypes.c_float()
        speed_u = ctypes.pointer(speed)

        tag = self.pDll.Get_Arm_Line_Speed(self.nSocket, speed_u)
        while tag and retry:
            logger_.info(f'Get_Arm_Line_Speed:{tag},retry is :{6 - retry}')
            tag = self.pDll.Get_Arm_Line_Speed(self.nSocket, speed_u)
            retry -= 1

        logger_.info(f'Get_Arm_Line_Speed:{tag}')
        return tag, speed.value

    def Get_Arm_Line_Acc(self, retry=0):
        """
        Get_Arm_Line_Acc 获取机械臂末端最大线加速度
        :return:
        """

        acc = ctypes.c_float()
        acc_u = ctypes.pointer(acc)

        tag = self.pDll.Get_Arm_Line_Acc(self.nSocket, acc_u)

        while tag and retry:
            logger_.info(f'Get_Arm_Line_Acc:{tag},retry is :{6 - retry}')
            tag = self.pDll.Get_Arm_Line_Acc(self.nSocket, acc_u)
            retry -= 1

        logger_.info(f'Get_Arm_Line_Acc:{tag}')
        return tag, acc.value

    def Get_Arm_Angular_Speed(self, retry=0):
        """
        Get_Arm_Angular_Speed 获取机械臂末端最大角速度
        :return:
        """

        speed = ctypes.c_float()
        speed_u = ctypes.pointer(speed)

        tag = self.pDll.Get_Arm_Angular_Speed(self.nSocket, speed_u)

        while tag and retry:
            logger_.info(f'Get_Arm_Angular_Speed:{tag},retry is :{6 - retry}')
            tag = self.pDll.Get_Arm_Angular_Speed(self.nSocket, speed_u)
            retry -= 1

        logger_.info(f'Get_Arm_Angular_Speed:{tag}')
        return tag, speed.value

    def Get_Arm_Angular_Acc(self, retry=0):
        """
        Get_Arm_Angular_Acc 获取机械臂末端最大角加速度
        :return:
        """

        acc = ctypes.c_float()
        acc_u = ctypes.pointer(acc)

        tag = self.pDll.Get_Arm_Angular_Acc(self.nSocket, acc_u)

        while tag and retry:
            logger_.info(f'Get_Arm_Angular_Acc:{tag},retry is :{6 - retry}')
            tag = self.pDll.Get_Arm_Angular_Acc(self.nSocket, acc_u)
            retry -= 1

        logger_.info(f'Get_Arm_Angular_Acc:{tag}')
        return tag, acc.value

    def Set_Arm_Tip_Init(self):
        # 设置机械臂末端参数为初始值
        tag = self.pDll.Set_Arm_Tip_Init(self.nSocket, 1)

        logger_.info(f'Set_Arm_Tip_Init:{tag}')
        logger_.info(f'设置机械臂末端参数为初始值')

        return tag

    def Set_Collision_Stage(self, stage, block=True):
        """
        Set_Collision_Stage 设置机械臂动力学碰撞检测等级
        :param stage: 等级：0~8，0-无碰撞，8-碰撞最灵敏
        :param block: RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:0-成功，失败返回:错误码, rm_define.h查询.
        """
        self.pDll.Set_Collision_Stage.argtypes = (
            ctypes.c_int, ctypes.c_int, ctypes.c_bool)
        self.pDll.Set_Collision_Stage.restype = self.check_error

        tag = self.pDll.Set_Collision_Stage(self.nSocket, stage, block)

        logger_.info(f'Set_Collision_Stage:{tag}')

        return tag

    def Get_Collision_Stage(self, retry=0):
        """
        Get_Collision_Stage 查询碰撞防护等级
        :return: 碰撞防护等级
        """
        self.pDll.Get_Collision_Stage.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        self.pDll.Get_Collision_Stage.restype = self.check_error

        stage = ctypes.c_int()
        stage_u = ctypes.pointer(stage)

        tag = self.pDll.Get_Collision_Stage(self.nSocket, stage_u)

        while tag and retry:
            logger_.info(f'Get_Collision_Stage:{tag},retry is :{6 - retry}')
            tag = self.pDll.Get_Collision_Stage(self.nSocket, stage_u)
            retry -= 1

        logger_.info(f'防撞等级是：{stage.value}')

        logger_.info(f'Get_Collision_Stage:{tag}')

        return tag, stage.value

    def Set_Joint_Zero_Offset(self, offset, block=True):
        """
        Set_Joint_Zero_Offset 该函数用于设置机械臂各关节零位补偿角度，一般在对机械臂零位进行标定后调用该函数
        :param offset: 关节1~6的零位补偿角度数组, 单位：度
        :param block: block RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:  0-成功，失败返回:错误码, rm_define.h查询.
        """
        le = self.code
        self.pDll.Set_Joint_Zero_Offset.argtypes = [
            ctypes.c_void_p, ctypes.c_float * le, ctypes.c_bool]
        self.pDll.Set_Joint_Zero_Offset.restype = self.check_error

        offset_arr = (ctypes.c_float * le)(*offset)

        tag = self.pDll.Set_Joint_Zero_Offset(self.nSocket, offset_arr, block)

        logger_.info(f'Set_Joint_Zero_Offset:{tag}')

        return tag


class Tool_Frame():
    def Auto_Set_Tool_Frame(self, point_num, block=True):
        """
        Auto_Set_Tool_Frame 六点法自动设置工具坐标系 标记点位
        :param point_num: 1~6代表6个标定点
        :param block: RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:
        """

        self.pDll.Auto_Set_Tool_Frame.argtypes = (
            ctypes.c_int, ctypes.c_byte, ctypes.c_bool)
        self.pDll.Auto_Set_Tool_Frame.restype = self.check_error

        tag = self.pDll.Auto_Set_Tool_Frame(self.nSocket, point_num, block)

        logger_.info(f'Auto_Set_Tool_Frame:{tag}')

        return tag

    def Generate_Auto_Tool_Frame(self, name, payload, x, y, z, block=True):
        """
        Generate_Auto_Tool_Frame 六点法自动设置工具坐标系 提交
        :param name: 工具坐标系名称，不能超过十个字节。
        :param payload: 新工具执行末端负载重量  单位kg
        :param x: 新工具执行末端负载位置 位置x 单位mm
        :param y: 新工具执行末端负载位置 位置y 单位mm
        :param z: 新工具执行末端负载位置 位置z 单位mm
        :param block: block RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:0-成功，失败返回:错误码, rm_define.h查询.

        """
        self.pDll.Generate_Auto_Tool_Frame.argtypes = (
            ctypes.c_int, ctypes.c_char_p, ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_bool)
        self.pDll.Generate_Auto_Tool_Frame.restype = self.check_error

        name = ctypes.c_char_p(name.encode('utf-8'))

        tag = self.pDll.Generate_Auto_Tool_Frame(
            self.nSocket, name, payload, x, y, z, block)

        logger_.info(f'Generate_Auto_Tool_Frame:{tag}')

        return tag

    def Manual_Set_Tool_Frame(self, name, pose, payload, x, y, z, block=True):
        """
        Manual_Set_Tool_Frame 手动设置工具坐标系
        :param name: 工具坐标系名称，不能超过十个字节
        :param pose: 新工具执行末端相对于机械臂法兰中心的位姿
        :param payload: 新工具执行末端负载重量  单位kg
        :param x: 新工具执行末端负载位置 位置x 单位m
        :param y: 新工具执行末端负载位置 位置y 单位m
        :param z: 新工具执行末端负载位置 位置z 单位m
        :param block: RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Manual_Set_Tool_Frame.argtypes = (
            ctypes.c_int, ctypes.c_char_p, Pose, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_bool)
        self.pDll.Manual_Set_Tool_Frame.restype = self.check_error

        name = ctypes.c_char_p(name.encode('utf-8'))

        pose1 = Pose()

        pose1.position = Pos(*pose[:3])
        pose1.euler = Euler(*pose[3:])

        tag = self.pDll.Manual_Set_Tool_Frame(
            self.nSocket, name, pose1, payload, x, y, z, block)

        logger_.info(f'Manual_Set_Tool_Frame:{tag}')

        return tag

    def Change_Tool_Frame(self, name, block=True):
        """
        Change_Tool_Frame 切换当前工具坐标系
        :param name: 目标工具坐标系名称
        :param block: RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return: 0-成功，失败返回:错误码, rm_define.h查询.

        """

        self.pDll.Change_Tool_Frame.argtypes = (
            ctypes.c_int, ctypes.c_char_p, ctypes.c_bool)
        self.pDll.Change_Tool_Frame.restype = self.check_error

        name = ctypes.c_char_p(name.encode('utf-8'))

        tag = self.pDll.Change_Tool_Frame(self.nSocket, name, block)

        logger_.info(f'Change_Tool_Frame:{tag}')

        return tag

    def Delete_Tool_Frame(self, name, block=True):
        """
        Delete_Tool_Frame 删除指定工具坐标系
        :param name: 要删除的工具坐标系名称
        :param block: RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        备注：删除坐标系后，机械臂将切换到机械臂法兰末端工具坐标系
        """

        self.pDll.Delete_Tool_Frame.argtypes = (
            ctypes.c_int, ctypes.c_char_p, ctypes.c_bool)
        self.pDll.Delete_Tool_Frame.restype = self.check_error

        name = ctypes.c_char_p(name.encode('utf-8'))

        tag = self.pDll.Delete_Tool_Frame(self.nSocket, name, block)

        logger_.info(f'Delete_Tool_Frame:{tag}')

        return tag

    def Update_Tool_Frame(self, name, pose, payload, x, y, z):
        """
        Update_Tool_Frame 修改指定工具坐标系
        :param name: 要修改的工具坐标系名称
        :param pose: 更新执行末端相对于机械臂法兰中心的位姿
        :param payload: 更新新工具执行末端负载重量  单位kg
        :param x: 更新工具执行末端负载位置 位置x 单位m
        :param y: 更新工具执行末端负载位置 位置y 单位m
        :param z: 更新工具执行末端负载位置 位置z 单位m
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Update_Tool_Frame.argtypes = (
            ctypes.c_int, ctypes.c_char_p, Pose, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float)
        self.pDll.Update_Tool_Frame.restype = self.check_error

        name = ctypes.c_char_p(name.encode('utf-8'))

        pose1 = Pose()

        pose1.position = Pos(*pose[:3])
        pose1.euler = Euler(*pose[3:])

        tag = self.pDll.Update_Tool_Frame(
            self.nSocket, name, pose1, payload, x, y, z)

        logger_.info(f'Update_Tool_Frame:{tag}')

        return tag

    def Set_Tool_Envelope(self, envelop_list: ToolEnvelopeList):
        """
        Set_Tool_Envelope 设置工具坐标系的包络参数
        :param envelop_list: 包络参数列表，每个工具最多支持 5 个包络球，可以没有包络
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """
        self.pDll.Set_Tool_Envelope.argtypes = (
            ctypes.c_int, ctypes.POINTER(ToolEnvelopeList))
        self.pDll.Set_Tool_Envelope.restype = self.check_error

        # tel_list = ToolEnvelopeList()

        tag = self.pDll.Set_Tool_Envelope(
            self.nSocket, ctypes.pointer(envelop_list))

        logger_.info(f'Set_Tool_Envelope:{tag}')

        return tag

    def Get_Tool_Envelope(self, tool_name) :
        """
        获取指定工具坐标系的包络参数
        :param tool_name: 指定工具坐标系名称
        :return:  0-成功，失败返回:错误码, rm_define.h查询.

        """

        self.pDll.Get_Tool_Envelope.argtypes = [
            ctypes.c_int, ctypes.c_char_p, ctypes.POINTER(ToolEnvelopeList)]
        self.pDll.Get_Tool_Envelope.restype = self.check_error

        tool_name = tool_name.encode("utf-8")
        tel_list = ToolEnvelopeList()
        tag = self.pDll.Get_Tool_Envelope(
            self.nSocket, tool_name, ctypes.pointer(tel_list))
        logger_.info(f'Get_Current_Tool_Frame:{tag}')

        return tag, tel_list.to_output()

    def Get_Current_Tool_Frame(self, retry=0):
        """
        Get_Current_Tool_Frame 获取当前工具坐标系
        :param tool:返回的坐标系
        :return:
        """

        self.pDll.Get_Current_Tool_Frame.argtypes = (
            ctypes.c_int, ctypes.POINTER(FRAME))
        self.pDll.Get_Current_Tool_Frame.restype = self.check_error

        frame = FRAME()

        tag = self.pDll.Get_Current_Tool_Frame(
            self.nSocket, ctypes.byref(frame))

        while tag and retry:
            logger_.info(
                f'Get_Current_Tool_Frame run failed :{tag},retry is :{6 - retry}')
            tag = self.pDll.Get_Current_Tool_Frame(
                self.nSocket, ctypes.byref(frame))

            retry -= 1

        logger_.info(f'Get_Current_Tool_Frame:{tag}')

        return tag, frame

    def Get_Given_Tool_Frame(self, name, retry=0):
        """
        Get_Given_Tool_Frame 获取指定工具坐标系
        :param name:指定的工具名称
        :param tool:返回的工具参数
        :return:
        """

        self.pDll.Get_Given_Tool_Frame.argtypes = (
            ctypes.c_int, ctypes.c_char_p, ctypes.POINTER(FRAME))

        self.pDll.Get_Given_Tool_Frame.restype = self.check_error

        name = ctypes.c_char_p(name.encode('utf-8'))
        frame = FRAME()

        tag = self.pDll.Get_Given_Tool_Frame(
            self.nSocket, name, ctypes.byref(frame))

        while tag and retry:
            logger_.info(
                f'Get_Given_Tool_Frame run failed :{tag},retry is :{6 - retry}')

            tag = self.pDll.Get_Given_Tool_Frame(
                self.nSocket, name, ctypes.byref(frame))

            retry -= 1

        logger_.info(f'Get_Given_Tool_Frame:{tag}')

        return tag, frame

    def Get_All_Tool_Frame(self, retry=0):
        """
        Get_All_Tool_Frame 获取所有工具坐标系名称
        :return:
        """

        self.pDll.Get_All_Tool_Frame.argtypes = (
            ctypes.c_int, ctypes.POINTER(FRAME_NAME), ctypes.POINTER(ctypes.c_int))

        self.pDll.Get_All_Tool_Frame.restype = self.check_error

        max_len = 10  # maximum number of tools

        names = (FRAME_NAME * max_len)()  # 创建 FRAME_NAME 数组
        names_ptr = ctypes.POINTER(FRAME_NAME)(names)  #

        len_ = ctypes.c_int()

        tag = self.pDll.Get_All_Tool_Frame(
            self.nSocket, names_ptr, ctypes.byref(len_))

        while tag and retry:
            logger_.info(
                f'Get_All_Tool_Frame run failed :{tag},retry is :{6 - retry}')
            tag = self.pDll.Get_All_Tool_Frame(
                self.nSocket, names_ptr, ctypes.byref(len_))
            retry -= 1

        logger_.info(f'Get_All_Tool_Frame:{tag}')

        tool_names = [names[i].name.decode('utf-8') for i in range(len_.value)]
        return tag, tool_names, len_.value


class Work_Frame():
    def Auto_Set_Work_Frame(self, name, point_num, block=True):
        """
        Auto_Set_Work_Frame 三点法自动设置工作坐标系
        :param name: 工作坐标系名称，不能超过十个字节。
        :param point_num: 1~3代表3个标定点，依次为原点、X轴一点、Y轴一点，4代表生成坐标系。
        :param block: 0-成功，失败返回:错误码, rm_define.h查询.
        :return:
        """

        self.pDll.Auto_Set_Work_Frame.argtypes = (
            ctypes.c_int, ctypes.c_char_p, ctypes.c_byte, ctypes.c_bool)
        self.pDll.Auto_Set_Work_Frame.restype = self.check_error

        name = ctypes.c_char_p(name.encode('utf-8'))
        tag = self.pDll.Auto_Set_Work_Frame(
            self.nSocket, name, point_num, block)

        logger_.info(f'Auto_Set_Work_Frame:{tag}')

        return tag

    def Manual_Set_Work_Frame(self, name, pose, block=True):
        """
        Manual_Set_Work_Frame 手动设置工作坐标系
        :param name: 工作坐标系名称，不能超过十个字节。
        :param pose: 新工作坐标系相对于基坐标系的位姿
        :param block:RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Manual_Set_Work_Frame.argtypes = (
            ctypes.c_int, ctypes.c_char_p, Pose, ctypes.c_bool)
        self.pDll.Manual_Set_Work_Frame.restype = self.check_error

        name = ctypes.c_char_p(name.encode('utf-8'))
        pose1 = Pose()

        pose1.position = Pos(*pose[:3])
        pose1.euler = Euler(*pose[3:])

        tag = self.pDll.Manual_Set_Work_Frame(self.nSocket, name, pose1, block)

        logger_.info(f'Manual_Set_Work_Fram:{tag}')

        return tag

    def Change_Work_Frame(self, name="Base"):
        """
        切换到某个工作坐标系，默认是base坐标系
        """

        self.pDll.Change_Work_Frame.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_bool]
        name = ctypes.c_char_p(name.encode('utf-8'))
        tag = self.pDll.Change_Work_Frame(self.nSocket, name, 1)
        logger_.info(f'Change_Work_Frame:{tag}')
        time.sleep(1)

        return tag

    def Delete_Work_Frame(self, name, block=True):
        """
        Delete_Work_Frame 删除指定工作坐标系
        :param name: 要删除的工具坐标系名称
        :param block: RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:0-成功，失败返回:错误码, rm_define.h查询.
        """
        self.pDll.Delete_Work_Frame.argtypes = (
            ctypes.c_int, ctypes.c_char_p, ctypes.c_bool)

        self.pDll.Delete_Work_Frame.restype = self.check_error

        name = ctypes.c_char_p(name.encode('utf-8'))

        tag = self.pDll.Delete_Work_Frame(self.nSocket, name, block)

        logger_.info(f'Delete_Work_Frame:{tag}')

        return tag

    def Update_Work_Frame(self, name, pose):
        """
        Update_Work_Frame 修改指定工作坐标系
        :param name: 要修改的工作坐标系名称
        :param pose: 更新工作坐标系相对于基坐标系的位姿
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Update_Work_Frame.argtypes = (
            ctypes.c_int, ctypes.c_char_p, Pose)
        self.pDll.Update_Work_Frame.restype = self.check_error

        name = ctypes.c_char_p(name.encode('utf-8'))

        pose1 = Pose()

        pose1.position = Pos(*pose[:3])
        pose1.euler = Euler(*pose[3:])

        tag = self.pDll.Update_Work_Frame(self.nSocket, name, pose1)

        logger_.info(f'Update_Work_Frame:{tag}')

        return tag

    def Get_Current_Work_Frame(self, retry=0):
        """
        Get_Current_Work_Frame 获取当前工作坐标系
        :return:
        """

        self.pDll.Get_Current_Work_Frame.argtypes = (
            ctypes.c_int, ctypes.POINTER(FRAME))

        self.pDll.Get_Current_Work_Frame.restype = self.check_error

        frame = FRAME()

        tag = self.pDll.Get_Current_Work_Frame(
            self.nSocket, ctypes.byref(frame))

        while tag and retry:
            logger_.info(
                f'Get_Current_Work_Frame run failed :{tag},retry is :{6 - retry}')
            tag = self.pDll.Get_Current_Work_Frame(
                self.nSocket, ctypes.byref(frame))

            retry -= 1

        logger_.info(f'Get_Current_Work_Frame:{tag}')

        return tag, frame

    def Get_Given_Work_Frame(self, name, retry=0):
        """
        Get_Given_Work_Frame 获取指定工作坐标系
        :return:指定工作坐标系得位姿
        """

        self.pDll.Get_Given_Work_Frame.argtypes = (
            ctypes.c_int, ctypes.c_char_p, ctypes.POINTER(Pose))
        self.pDll.Get_Given_Work_Frame.restype = self.check_error

        name = ctypes.c_char_p(name.encode('utf-8'))

        pose = Pose()

        tag = self.pDll.Get_Given_Work_Frame(
            self.nSocket, name, ctypes.byref(pose))

        while tag and retry:
            logger_.info(
                f'Get_Given_Work_Frame run failed :{tag},retry is :{6 - retry}')

            tag = self.pDll.Get_Given_Work_Frame(
                self.nSocket, name, ctypes.byref(pose))

            retry -= 1

        logger_.info(f'Get_Given_Work_Frame:{tag}')

        position = pose.position
        euler = pose.euler
        return tag, [position.x, position.y, position.z, euler.rx, euler.ry, euler.rz]

    def Get_All_Work_Frame(self, retry=0):
        """
        Get_All_Work_Frame 获取所有工作坐标系名称
        :return:
        """

        self.pDll.Get_All_Work_Frame.argtypes = (
            ctypes.c_int, ctypes.POINTER(FRAME_NAME), ctypes.POINTER(ctypes.c_int))

        max_len = 10  # maximum number of tools
        names = (FRAME_NAME * max_len)()  # creates an array of FRAME_NAME
        names_ptr = ctypes.POINTER(FRAME_NAME)(names)  #
        len_ = ctypes.c_int()

        tag = self.pDll.Get_All_Work_Frame(
            self.nSocket, names_ptr, ctypes.byref(len_))

        while tag and retry:
            logger_.info(
                f'Get_All_Work_Frame run failed :{tag},retry is :{6 - retry}')
            tag = self.pDll.Get_All_Work_Frame(
                self.nSocket, names_ptr, ctypes.byref(len_))
            retry -= 1

        logger_.info(f'Get_All_Work_Frame:{tag}')

        job_names = [names[i].name.decode('utf-8') for i in range(len_.value)]
        return tag, job_names, len_.value


class Arm_State():
    def Get_Current_Arm_State(self, retry=0):
        """Gets the arm's current states. Returns 0 iff success.
        Only works with POSE but not POSE_c, i.e., doesn't return quaternion.
        Use forward_kinematics() instead if quaternion is a must."""

        le = self.code

        self.pDll.Get_Current_Arm_State.argtypes = (ctypes.c_int, ctypes.c_float * le, ctypes.POINTER(Pose),
                                                    ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16))
        self.pDll.Get_Current_Arm_State.restype = self.check_error
        joints = (ctypes.c_float * le)()
        curr_pose = Pose()
        cp_ptr = ctypes.pointer(curr_pose)
        arm_err_ptr = ctypes.pointer(ctypes.c_uint16())
        sys_err_ptr = ctypes.pointer(ctypes.c_uint16())
        error_code = self.pDll.Get_Current_Arm_State(
            self.nSocket, joints, cp_ptr, arm_err_ptr, sys_err_ptr)
        while error_code and retry:
            # sleep(0.3)
            logger_.warning(
                f"Failed to get curr arm states. Error Code: {error_code}\tRetry Count: {retry}")
            error_code = self.pDll.Get_Current_Arm_State(
                self.nSocket, joints, cp_ptr, arm_err_ptr, sys_err_ptr)
            retry -= 1

        logger_.info(f'Get_Current_Arm_State:{error_code}')

        position = curr_pose.position
        euler = curr_pose.euler
        curr_pose = [position.x, position.y,
                     position.z, euler.rx, euler.ry, euler.rz]
        return error_code, list(joints), curr_pose, arm_err_ptr.contents.value, sys_err_ptr.contents.value

    def Get_Joint_Temperature(self):
        """
        Get_Joint_Temperature 获取关节当前温度
        :return:
        """

        le = self.code

        self.pDll.Get_Joint_Temperature.argtypes = (
            ctypes.c_int, ctypes.c_float * le)

        self.pDll.Get_Joint_Temperature.restype = self.check_error

        temperature = (ctypes.c_float * le)()

        tag = self.pDll.Get_Joint_Temperature(self.nSocket, temperature)

        logger_.info(f'Get_Joint_Temperature:{tag}')

        return tag, list(temperature)

    def Get_Joint_Current(self):
        """
        Get_Joint_Current 获取关节当前电流
        :return:
        """
        le = self.code

        self.pDll.Get_Joint_Current.argtypes = (
            ctypes.c_int, ctypes.c_float * le)

        self.pDll.Get_Joint_Current.restype = self.check_error

        current = (ctypes.c_float * le)()

        tag = self.pDll.Get_Joint_Current(self.nSocket, current)

        logger_.info(f'Get_Joint_Current:{tag}')

        return tag, list(current)

    def Get_Joint_Voltage(self):
        """
        Get_Joint_Voltage 获取关节当前电压
        :return:
        """
        le = self.code

        self.pDll.Get_Joint_Voltage.argtypes = (
            ctypes.c_int, ctypes.c_float * le)

        self.pDll.Get_Joint_Voltage.restype = self.check_error

        voltage = (ctypes.c_float * le)()

        tag = self.pDll.Get_Joint_Voltage(self.nSocket, voltage)

        logger_.info(f'Get_Joint_Voltage:{tag}')

        return tag, list(voltage)

    def Get_Joint_Degree(self):
        """
        Get_Joint_Degree 获取关节当前电压
        :return:
        """

        self.pDll.Get_Joint_Degree.argtypes = (
            ctypes.c_int, ctypes.c_float * 7)

        self.pDll.Get_Joint_Degree.restype = self.check_error

        joint = (ctypes.c_float * 7)()

        tag = self.pDll.Get_Joint_Degree(self.nSocket, joint)

        logger_.info(f'Get_Joint_Degree:{tag}')

        return tag, list(joint)

    def Get_Arm_All_State(self, retry=0):
        """
        Get_Arm_All_State 获取机械臂所有状态信息
        :param self:
        :return:
        """
        self.pDll.Get_Arm_All_State.argtypes = (
            ctypes.c_int, ctypes.POINTER(JOINT_STATE))
        self.pDll.Get_Arm_All_State.restype = self.check_error

        joint_status = JOINT_STATE()

        # joint_status_p = ctypes.pointer(joint_status)
        tag = self.pDll.Get_Arm_All_State(self.nSocket, joint_status)

        while tag and retry:
            logger_.info(f'Get_Arm_All_State:{tag},retry is :{6 - retry}')
            tag = self.pDll.Get_Arm_All_State(self.nSocket, joint_status)
            retry -= 1

        logger_.info(f'Get_Arm_All_State:{tag}')

        return tag, joint_status

    def Get_Arm_Plan_Num(self, retry=0):
        """
        Get_Arm_Plan_Num    查询规划计数
        :return:
        """

        self.pDll.Get_Arm_Plan_Num.argtypes = (
            ctypes.c_int, ctypes.POINTER(ctypes.c_int))
        self.pDll.Get_Arm_Plan_Num.restype = self.check_error

        plan_num = ctypes.c_int()
        plan_num_p = ctypes.pointer(plan_num)

        tag = self.pDll.Get_Arm_Plan_Num(self.nSocket, plan_num_p)

        while tag and retry:
            logger_.info(f'Get_Arm_Plan_Num:{tag},retry is :{6 - retry}')
            tag = self.pDll.Get_Arm_Plan_Num(self.nSocket, plan_num_p)

            retry -= 1

        logger_.info(f'Get_Arm_Plan_Num:{tag}')

        return tag, plan_num.value


class Initial_Pose():
    def Set_Arm_Init_Pose(self, target, block=True):
        """
        Set_Arm_Init_Pose 设置机械臂的初始位置角度
        :param target: 机械臂初始位置关节角度数组
        :param block: RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:
        """

        if self.code == 6:
            self.pDll.Set_Arm_Init_Pose.argtypes = (
                ctypes.c_int, ctypes.c_float * 6, ctypes.c_bool)
            self.pDll.Set_Arm_Init_Pose.restype = self.check_error

            target = (ctypes.c_float * 6)(*target)

            tag = self.pDll.Set_Arm_Init_Pose(self.nSocket, target, block)

        else:
            self.pDll.Set_Arm_Init_Pose.argtypes = (
                ctypes.c_int, ctypes.c_float * 7, ctypes.c_bool)
            self.pDll.Set_Arm_Init_Pose.restype = self.check_error

            target = (ctypes.c_float * 7)(*target)

            tag = self.pDll.Set_Arm_Init_Pose(self.nSocket, target, block)

        logger_.info(f'Set_Arm_Init_Pose:{tag}')
        return tag

    def Get_Arm_Init_Pose(self):
        """
        Set_Arm_Init_Pose 获取机械臂初始位置角度
        :return:joint 机械臂初始位置关节角度数组
        """

        if self.code == 6:
            self.pDll.Get_Arm_Init_Pose.argtypes = (
                ctypes.c_int, ctypes.c_float * 6)
            self.pDll.Get_Arm_Init_Pose.restype = self.check_error

            target = (ctypes.c_float * 6)()

            tag = self.pDll.Get_Arm_Init_Pose(self.nSocket, target)

        else:
            self.pDll.Get_Arm_Init_Pose.argtypes = (
                ctypes.c_int, ctypes.c_float * 7)
            self.pDll.Get_Arm_Init_Pose.restype = self.check_error

            target = (ctypes.c_float * 7)()

            tag = self.pDll.Get_Arm_Init_Pose(self.nSocket, target)

        logger_.info(f'Get_Arm_Init_Pose:{tag}')

        return tag, list(target)

    def Set_Install_Pose(self, x, y, z, block=True):
        """
        Set_Install_Pose     设置安装方式参数

        :param x: 旋转角 单位 °
        :param y: 俯仰角 单位 °
        :param z: 方位角 单位 °
        :param block: RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Set_Install_Pose.argtypes = (
            ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_bool)
        self.pDll.Set_Install_Pose.restype = self.check_error

        tag = self.pDll.Set_Install_Pose(self.nSocket, x, y, z, block)

        logger_.info(f'Set_Install_Pose:{tag}')

        return tag

    def Get_Install_Pose(self):
        self.pDll.Get_Install_Pose.argtypes = (ctypes.c_int,
                                               ctypes.POINTER(ctypes.c_float), ctypes.POINTER(
                                                   ctypes.c_float),
                                               ctypes.POINTER(ctypes.c_float))
        x = ctypes.c_float()
        y = ctypes.c_float()
        z = ctypes.c_float()
        tag = self.pDll.Get_Install_Pose(self.nSocket, x, y, z)
        logger_.info(f'Get_Install_Pose:{tag}')

        return tag, x, y, z


class Move_Plan:
    def Movej_Cmd(self, joint, v, trajectory_connect=0, r=0, block=True):
        """
       Movej_Cmd 关节空间运动
       ArmSocket socket句柄
       joint 目标关节1~7角度数组
       v 速度百分比系数，1~100。
       r 交融半径百分比系数，0~100。
       trajectory_connect 代表是否和下一条运动一起规划，0代表立即规划，1代表和下一条轨迹一起规划，当为1时，轨迹不会立即执行
       block True 阻塞 False 非阻塞
       return 0-成功，失败返回:错误码, rm_define.h查询.
        """

        le = self.code
        float_joint = ctypes.c_float * le
        joint = float_joint(*joint)
        self.pDll.Movej_Cmd.argtypes = (ctypes.c_int, ctypes.c_float * le, ctypes.c_byte,
                                        ctypes.c_float, ctypes.c_int, ctypes.c_bool)

        self.pDll.Movej_Cmd.restype = self.check_error

        tag = self.pDll.Movej_Cmd(
            self.nSocket, joint, v, r, trajectory_connect, block)
        logger_.info(f'Movej_Cmd:{tag}')

        return tag

    def Movel_Cmd(self, pose, v, trajectory_connect=0, r=0, block=True):
        """
        笛卡尔空间直线运动

        pose 目标位姿,位置单位：米，姿态单位：弧度
        v 速度百分比系数，1~100。
        r 交融半径百分比系数，0~100。
        trajectory_connect 代表是否和下一条运动一起规划，0代表立即规划，1代表和下一条轨迹一起规划，当为1时，轨迹不会立即执行
        block True 阻塞 False 非阻塞

       return:0-成功，失败返回:错误码, rm_define.h查询
        """

        po1 = Pose()
        po1.position = Pos(*pose[:3])
        po1.euler = Euler(*pose[3:])

        self.pDll.Movel_Cmd.argtypes = (
            ctypes.c_int, Pose, ctypes.c_byte, ctypes.c_float, ctypes.c_int, ctypes.c_bool)
        self.pDll.Movel_Cmd.restype = self.check_error
        tag = self.pDll.Movel_Cmd(
            self.nSocket, po1, v, r, trajectory_connect, block)
        logger_.info(f'Movel_Cmd:{tag}')

        return tag

    def Movec_Cmd(self, pose_via, pose_to, v, loop, trajectory_connect=0, r=0, block=True):
        """
        Movec_Cmd 笛卡尔空间圆弧运动
        :param pose_via: 中间点位姿，位置单位：米，姿态单位：弧度
        :param pose_to: 终点位姿
        :param v 速度百分比系数，1~100。
        :param r 交融半径百分比系数，0~100。
        :param trajectory_connect: 代表是否和下一条运动一起规划，0代表立即规划，1代表和下一条轨迹一起规划，当为1时，轨迹不会立即执行
        :param loop:规划圈数，目前默认0.
        :param block:RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待机械臂到达位置或者规划失败
        :return:
        """

        self.pDll.Movec_Cmd.argtypes = (
            ctypes.c_int, Pose, Pose, ctypes.c_byte, ctypes.c_float, ctypes.c_byte, ctypes.c_int, ctypes.c_bool)
        self.pDll.Movec_Cmd.restype = self.check_error

        pose1 = Pose()

        pose1.position = Pos(*pose_via[:3])
        pose1.euler = Euler(*pose_via[3:])

        pose2 = Pose()

        pose2.position = Pos(*pose_to[:3])
        pose2.euler = Euler(*pose_to[3:])

        tag = self.pDll.Movec_Cmd(
            self.nSocket, pose1, pose2, v, r, loop, trajectory_connect, block)

        logger_.info(f'Movec_Cmd:{tag}')

        return tag

    def Movej_P_Cmd(self, pose, v, trajectory_connect=0, r=0, block=True):
        """
        该函数用于关节空间运动到目标位姿
        param ArmSocket socket句柄
        pose: 目标位姿，位置单位：米，姿态单位：弧度。 注意：目标位姿必须是机械臂当前工具坐标系相对于当前工作坐标系的位姿，
              用户在使用该指令前务必确保，否则目标位姿会出错！！
        v 速度百分比系数，1~100。
        r 交融半径百分比系数，0~100。
        trajectory_connect: 代表是否和下一条运动一起规划，0代表立即规划，1代表和下一条轨迹一起规划，当为1时，轨迹不会立即执行
        block True 阻塞 False 非阻塞
        return 0-成功，失败返回:错误码

        """
        po1 = Pose()

        po1.position = Pos(*pose[:3])
        po1.euler = Euler(*pose[3:])

        self.pDll.Movej_P_Cmd.argtypes = (
            ctypes.c_int, Pose, ctypes.c_byte, ctypes.c_float, ctypes.c_int, ctypes.c_bool)
        self.pDll.Movej_P_Cmd.restype = self.check_error

        tag = self.pDll.Movej_P_Cmd(
            self.nSocket, po1, v, r, trajectory_connect, block)
        logger_.info(f'Movej_P_Cmd执行结果:{tag}')

        return tag

    def Moves_Cmd(self, pose, v, trajectory_connect=0, r=0, block=True):
        """
        该函数用于样条曲线运动
        :param ArmSocket socket句柄
        :param pose: 目标位姿，位置单位：米，姿态单位：弧度。 注意：目标位姿必须是机械臂当前工具坐标系相对于当前工作坐标系的位姿，
              用户在使用该指令前务必确保，否则目标位姿会出错！！
        :param v 速度百分比系数，1~100。
        :param r 交融半径百分比系数，0~100。
        :param trajectory_connect: 代表是否和下一条运动一起规划，0代表立即规划，1代表和下一条轨迹一起规划，当为1时，轨迹不会立即执行
        :param block True 阻塞 False 非阻塞
        :return 0-成功，失败返回:错误码

        """
        po1 = Pose()

        po1.position = Pos(*pose[:3])
        po1.euler = Euler(*pose[3:])

        self.pDll.Moves_Cmd.argtypes = (
            ctypes.c_int, Pose, ctypes.c_byte, ctypes.c_float, ctypes.c_int, ctypes.c_bool)
        self.pDll.Moves_Cmd.restype = self.check_error

        tag = self.pDll.Moves_Cmd(
            self.nSocket, po1, v, r, trajectory_connect, block)
        logger_.info(f'Moves_Cmd执行结果:{tag}')

        return tag

    def Movej_CANFD(self, joint, follow, expand=0):
        """
        Movej_CANFD 角度不经规划，直接通过CANFD透传给机械臂
        :param joint: 关节1~7目标角度数组
        :param follow: 是否高跟随
        因此只要控制器运行正常并且目标角度在可达范围内，机械臂立即返回成功指令，此时机械臂可能仍在运行；
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """

        if self.code == 6:

            self.pDll.Movej_CANFD.argtypes = (
                ctypes.c_int, ctypes.c_float * 6, ctypes.c_bool, ctypes.c_float)
            self.pDll.Movej_CANFD.restype = self.check_error

            joints = (ctypes.c_float * 6)(*joint)

        else:
            self.pDll.Movej_CANFD.argtypes = (
                ctypes.c_int, ctypes.c_float * 7, ctypes.c_bool, ctypes.c_float)
            self.pDll.Movej_CANFD.restype = self.check_error

            joints = (ctypes.c_float * 7)(*joint)

        tag = self.pDll.Movej_CANFD(self.nSocket, joints, follow, expand)

        return tag

    def Movep_CANFD(self, pose, follow):
        """
        Movep_CANFD 位资不经规划，直接通过CANFD透传给机械臂
        :param pose: 关节1~7目标角度数组
        :param follow: 是否高跟随
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """
        if len(pose) > 6:
            po1 = Pose()
            po1.position = Pos(*pose[:3])
            po1.quaternion = Quat(*pose[3:])
        else:
            po1 = Pose()
            po1.position = Pos(*pose[:3])
            po1.euler = Euler(*pose[3:])

        self.pDll.Movep_CANFD.argtypes = (ctypes.c_int, Pose, ctypes.c_bool)
        self.pDll.Movep_CANFD.restype = self.check_error
        tag = self.pDll.Movep_CANFD(self.nSocket, po1, follow)

        return tag

    def MoveRotate_Cmd(self, rotateAxis, rotateAngle, choose_axis, v, trajectory_connect=0, r=0, block=True):
        """
        MoveRotate_Cmd  计算环绕运动位姿并按照结果运动
        :param rotateAxis:旋转轴: 1:x轴, 2:y轴, 3:z轴
        :param rotateAngle:旋转角度: 旋转角度, 单位(度)
        :param choose_axis:指定计算时使用的坐标系
        :param v 速度百分比系数，1~100。
        :param trajectory_connect:代表是否和下一条运动一起规划，0代表立即规划，1代表和下一条轨迹一起规划，当为1时，轨迹不会立即执行
        :param r 交融半径百分比系数，0~100。
        :param block:RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞
        :return:0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.MoveRotate_Cmd.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_float, Pose, ctypes.c_byte,
                                             ctypes.c_float, ctypes.c_int, ctypes.c_bool)

        self.pDll.MoveRotate_Cmd.restype = self.check_error

        pose = Pose()

        pose.position = Pos(*choose_axis[:3])
        pose.euler = Euler(*choose_axis[3:])

        tag = self.pDll.MoveRotate_Cmd(
            self.nSocket, rotateAxis, rotateAngle, pose, v, r, trajectory_connect, block)

        logger_.info(f'MoveRotate_Cmd:{tag}')

        return tag

    def MoveCartesianTool_Cmd(self, joint_cur, movelengthx, movelengthy, movelengthz, m_dev, v, trajectory_connect=0,
                              r=0,
                              block=True):
        """
        cartesian_tool           沿工具端位姿移动
        :param joint_cur: 当前关节角度
        :param movelengthx: 沿X轴移动长度，米为单位
        :param movelengthy: 沿Y轴移动长度，米为单位
        :param movelengthz: 沿Z轴移动长度，米为单位
        :param m_dev: 机械臂型号
        :param v 速度百分比系数，1~100。
        :param trajectory_connect: 代表是否和下一条运动一起规划，0代表立即规划，1代表和下一条轨迹一起规划，当为1时，轨迹不会立即执行
        :param r 交融半径百分比系数，0~100。
        :param block:RM_NONBLOCK-非阻塞，发送后立即返回; RM_BLOCK-阻塞，等待机械臂到达位置或者规划失败
        :return:
        """

        if self.code == 6:

            self.pDll.MoveCartesianTool_Cmd.argtypes = (
                ctypes.c_int, ctypes.c_float *
                6, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int,
                ctypes.c_byte, ctypes.c_float, ctypes.c_int, ctypes.c_bool)
            self.pDll.MoveCartesianTool_Cmd.restype = self.check_error

            joints = (ctypes.c_float * 6)(*joint_cur)

        else:

            self.pDll.MoveCartesianTool_Cmd.argtypes = (
                ctypes.c_int, ctypes.c_float *
                7, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int,
                ctypes.c_byte, ctypes.c_float, ctypes.c_int, ctypes.c_bool)
            self.pDll.MoveCartesianTool_Cmd.restype = self.check_error

            joints = (ctypes.c_float * 7)(*joint_cur)

        tag = self.pDll.MoveCartesianTool_Cmd(self.nSocket, joints, movelengthx, movelengthy, movelengthz, m_dev, v, r,
                                              trajectory_connect, block)

        logger_.info(f'MoveCartesianTool_Cmd:{tag}')

        return tag

    def Get_Current_Trajectory(self):
        """
        Get_Current_Trajectory 获取当前轨迹规划类型

        :return:
            tuple[int, int, list[float]]: 一个包含三个元素的元组，分别表示：  
            - int: 0-成功，失败返回:错误码, errro_message查询.。  
            - int: 轨迹规划类型（由 ARM_CTRL_MODES 枚举定义的值）。  
            - list[float]: 包含7个浮点数的列表，关节规划及无规划时，该列表为关节角度数组；其他类型为末端位姿数组[x,y,z,rx,ry,rz]。  
        """

        self.pDll.Get_Current_Trajectory.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                                                     ctypes.c_float * 7]
        self.pDll.Get_Current_Trajectory.restype = self.check_error

        type = ctypes.c_int()
        data = (ctypes.c_float * 7)()
        tag = self.pDll.Get_Current_Trajectory(
            self.nSocket, ctypes.byref(type), data)

        logger_.info(f'Get_Current_Trajectory result:{tag}')
        return tag, type.value, list(data)

    def Move_Stop_Cmd(self, block=True):
        """
         Move_Stop_Cmd 突发状况 机械臂以最快速度急停，轨迹不可恢复
        :param block:RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Move_Stop_Cmd.argtypes = (ctypes.c_int, ctypes.c_bool)
        self.pDll.Move_Stop_Cmd.restype = self.check_error

        tag = self.pDll.Move_Stop_Cmd(self.nSocket, block)

        logger_.info(f'Move_Stop_Cmd:{tag}')

        return tag

    def Move_Pause_Cmd(self, block=True):
        """
         Move_Pause_Cmd 轨迹暂停，暂停在规划轨迹上，轨迹可恢复
        :param block:RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Move_Pause_Cmd.argtypes = (ctypes.c_int, ctypes.c_bool)
        self.pDll.Move_Pause_Cmd.restype = self.check_error

        tag = self.pDll.Move_Pause_Cmd(self.nSocket, block)

        logger_.info(f'Move_Pause_Cmd:{tag}')

        return tag

    def Move_Continue_Cmd(self, block=True):
        """
         Move_Continue_Cmd 轨迹暂停后，继续当前轨迹运动
        :param block:RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Move_Continue_Cmd.argtypes = (ctypes.c_int, ctypes.c_bool)
        self.pDll.Move_Continue_Cmd.restype = self.check_error

        tag = self.pDll.Move_Continue_Cmd(self.nSocket, block)

        logger_.info(f'Move_Continue_Cmd:{tag}')

        return tag

    def Clear_Current_Trajectory(self, block=True):
        """
         Clear_Current_Trajectory 清除当前轨迹，必须在暂停后使用，否则机械臂会发生意外！！！！
        :param block:RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Clear_Current_Trajectory.argtypes = (
            ctypes.c_int, ctypes.c_bool)
        self.pDll.Clear_Current_Trajectory.restype = self.check_error

        tag = self.pDll.Clear_Current_Trajectory(self.nSocket, block)

        logger_.info(f'Clear_Current_Trajectory:{tag}')

        return tag

    def Clear_All_Trajectory(self, block=True):
        """
         Clear_All_Trajectory 清除所有轨迹，必须在暂停后使用，否则机械臂会发生意外！！！！
        :param block:RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Clear_All_Trajectory.argtypes = (ctypes.c_int, ctypes.c_bool)
        self.pDll.Clear_All_Trajectory.restype = self.check_error

        tag = self.pDll.Clear_All_Trajectory(self.nSocket, block)

        logger_.info(f'Clear_All_Trajectory:{tag}')

        return tag


class Teaching:
    def Joint_Teach_Cmd(self, num, direction, v, block=True):
        """
        Joint_Teach_Cmd 关节示教
        :param num: 示教关节的序号，1~7
        :param direction: 示教方向，0-负方向，1-正方向
        :param v: 速度比例1~100，即规划速度和加速度占关节最大线转速和加速度的百分比
        :param block:
        :return:
        """

        self.pDll.Joint_Teach_Cmd.argtypes = (
            ctypes.c_int, ctypes.c_byte, ctypes.c_byte, ctypes.c_byte, ctypes.c_bool)
        self.pDll.Joint_Teach_Cmd.restype = self.check_error

        tag = self.pDll.Joint_Teach_Cmd(self.nSocket, num, direction, v, block)

        logger_.info(f'Joint_Teach_Cmd:{tag}')

        return tag

    def Joint_Step_Cmd(self, num, step, v, block=True):
        """
        Joint_Step_Cmd 关节步进
        :param num: 关节序号，1~7
        :param step: 步进的角度
        :param v: 速度比例1~100，即规划速度和加速度占机械臂末端最大线速度和线加速度的百分比
        :param block: RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待机械臂返回失败或者到达位置指令
        :return:  0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Joint_Step_Cmd.argtypes = (
            ctypes.c_int, ctypes.c_byte, ctypes.c_float, ctypes.c_byte, ctypes.c_bool)

        self.pDll.Joint_Step_Cmd.restype = self.check_error

        tag = self.pDll.Joint_Step_Cmd(self.nSocket, num, step, v, block)

        logger_.info(f'Joint_Step_Cmd:{tag}')

        return tag

    def Ort_Step_Cmd(self, type, step, v, block=True):
        """
        Ort_Step_Cmd 当前工作坐标系下，姿态步进
        :param type:示教类型 0:RX 1:RY 2:RZ
        :param step:步进的弧度，单位rad，精确到0.001rad
        :param v:速度比例1~100，即规划速度和加速度占机械臂末端最大线速度和线加速度的百分比
        :param block:RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待机械臂返回失败或者到达位置指令
        :return:
        """

        self.pDll.Ort_Step_Cmd.argtypes = (
            ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_byte, ctypes.c_bool)
        self.pDll.Ort_Step_Cmd.restype = self.check_error

        tag = self.pDll.Ort_Step_Cmd(self.nSocket, type, step, v, block)

        logger_.info(f'Ort_Step_Cmd:{tag}')

        return tag

    def Pos_Teach_Cmd(self, type, direction, v, block=True):
        """
        Pos_Teach_Cmd 当前工作坐标系下，笛卡尔空间位置示教
        :param type:示教类型 0:x轴方向  1：y轴方向 2：z轴方向
        :param direction:示教方向，0-负方向，1-正方向
        :param v:速度比例1~100，即规划速度和加速度占机械臂末端最大线速度和线加速度的百分比
        :param block:RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Pos_Teach_Cmd.argtypes = (
            ctypes.c_int, ctypes.c_int, ctypes.c_byte, ctypes.c_byte, ctypes.c_bool)
        self.pDll.Pos_Teach_Cmd.restype = self.check_error

        tag = self.pDll.Pos_Teach_Cmd(self.nSocket, type, direction, v, block)

        logger_.info(f'Pos_Teach_Cmd:{tag}')

        return tag

    def Pos_Step_Cmd(self, type_, step, v, block=True):
        """
        Pos_Step_Cmd 当前工作坐标系下，位置步进
        ArmSocket socket句柄
        type 示教类型 x:0 y:1 z:2
        step 步进的距离，单位m，精确到0.001mm
        v 速度比例1~100，即规划速度和加速度占机械臂末端最大线速度和线加速度的百分比
        block RM_NONBLOCK-非阻塞，发送后立即返回; RM_BLOCK-阻塞，等待机械臂返回失败或者到达位置指令


        return 0-成功，失败返回:错误码, rm_define.h查询.
        """

        if type_ == 0:
            type_ = POS_TEACH_MODES.X_Dir
        elif type_ == 1:
            type_ = POS_TEACH_MODES.Y_Dir
        elif type_ == 2:
            type_ = POS_TEACH_MODES.Z_Dir

        self.pDll.Pos_Step_Cmd.argtypes = (
            ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_byte, ctypes.c_bool)
        self.pDll.Pos_Step_Cmd.restype = self.check_error
        tag = self.pDll.Pos_Step_Cmd(self.nSocket, type_, step, v, block)
        logger_.info(f'Pos_Step_Cmd: {tag}')
        return tag

    def Ort_Teach_Cmd(self, type, direction, v, block=True):
        """

        :param type:
            0, // RX轴方向
            1, // RY轴方向
            2, // RZ轴方向
        :param direction: 示教方向，0-负方向，1-正方向
        :param v: 速度比例1~100，即规划速度和加速度占机械臂末端最大角速度和角加速度的百分比
        :param block: RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:
        """

        self.pDll.Ort_Teach_Cmd.argtypes = (
            ctypes.c_int, ctypes.c_int, ctypes.c_byte, ctypes.c_byte, ctypes.c_bool)
        self.pDll.Ort_Teach_Cmd.restype = self.check_error

        tag = self.pDll.Ort_Teach_Cmd(self.nSocket, type, direction, v, block)

        logger_.info(f'Ort_Teach_Cmd:{tag}')

        return tag

    def Teach_Stop_Cmd(self, block=True):
        """
        Teach_Stop_Cmd 示教停止
        :param block:RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Teach_Stop_Cmd.argtypes = (ctypes.c_int, ctypes.c_bool)
        self.pDll.Teach_Stop_Cmd.restype = self.check_error

        tag = self.pDll.Teach_Stop_Cmd(self.nSocket, block)

        logger_.info(f'Teach_Stop_Cmd:{tag}')

        return tag

    def Set_Teach_Frame(self, type, block=True):
        """
        Set_Teach_Frame      切换示教运动坐标系
        :param type: 0: 基座标运动, 1: 工具坐标系运动
        :param block: RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Set_Teach_Frame.argtypes = (
            ctypes.c_int, ctypes.c_int, ctypes.c_int)

        self.pDll.Set_Teach_Frame.restype = self.check_error

        tag = self.pDll.Set_Teach_Frame(self.nSocket, type, block)
        logger_.info(f'Set_Teach_Frame:{tag}')

        return tag

    def Get_Teach_Frame(self):
        """
        Get_Teach_Frame      获取示教参考坐标系
        :return: type: 0: 基座标运动, 1: 工具坐标系运动
        """

        self.pDll.Get_Teach_Frame.argtypes = (
            ctypes.c_int, ctypes.POINTER(ctypes.c_int))

        self.pDll.Get_Teach_Frame.restype = self.check_error

        type = ctypes.c_int()
        tag = self.pDll.Get_Teach_Frame(self.nSocket, ctypes.byref(type))
        logger_.info(f'Get_Teach_Frame:{tag}')

        return tag, type.value


class Set_controller():

    def Get_Controller_State(self, retry=0):
        """
        Get_Controller_State 获取控制器状态
        :return:电压,电流,温度
        """

        self.pDll.Get_Controller_State.argtypes = (
            ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float
                                                                         ), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint16))
        self.pDll.Get_Controller_State.restype = self.check_error
        voltage = ctypes.c_float()
        current = ctypes.c_float()
        temperature = ctypes.c_float()
        sys_err = ctypes.c_uint16()

        tag = self.pDll.Get_Controller_State(self.nSocket, ctypes.byref(voltage), ctypes.byref(current),
                                             ctypes.byref(temperature
                                                          ), ctypes.byref(sys_err))

        while tag and retry:
            logger_.info(f'Get_Controller_State:{tag},retry is :{6 - retry}')
            tag = self.pDll.Get_Controller_State(self.nSocket, ctypes.byref(voltage), ctypes.byref(current),
                                                 ctypes.byref(temperature
                                                              ), ctypes.byref(sys_err))

            retry -= 1

        return tag, voltage.value, current.value, temperature.value

    def Set_WiFi_AP_Data(self, wifi_name, password):
        """
        Set_WiFi_AP_Data 开启控制器WiFi AP模式设置
        :param wifi_name: 控制器wifi名称
        :param password: wifi密码
        :return: 返回值：0-成功，失败返回:错误码, rm_define.h查询.
        非阻塞模式，下发后，机械臂进入WIFI AP通讯模式
        """

        self.pDll.Set_WiFi_AP_Data.argytypes = (
            ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p)
        self.pDll.Set_WiFi_AP_Data.restype = self.check_error

        wifi_name = ctypes.c_char_p(wifi_name.encode('utf-8'))
        password = ctypes.c_char_p(password.encode('utf-8'))

        tag = self.pDll.Set_WiFi_AP_Data(self.nSocket, wifi_name, password)

        logger_.info(f'Set_WiFi_AP_Data:{tag}')

        return tag

    def Set_WiFI_STA_Data(self, router_name, password):
        """
        Set_WiFI_STA_Data 控制器WiFi STA模式设置
        :param router_name: 路由器名称
        :param password: 路由器Wifi密码
        :return: 返回值：0-成功，失败返回:错误码, rm_define.h查询.
        非阻塞模式：设置成功后，机械臂进入WIFI STA通信模式        """

        self.pDll.Set_WiFI_STA_Data.argytypes = (
            ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p)
        self.pDll.Set_WiFI_STA_Data.restype = self.check_error

        router_name = ctypes.c_char_p(router_name.encode('utf-8'))
        password = ctypes.c_char_p(password.encode('utf-8'))

        tag = self.pDll.Set_WiFI_STA_Data(self.nSocket, router_name, password)

        logger_.info(f'Set_WiFI_STA_Data:{tag}')

        return tag

    def Set_USB_Data(self, baudrate):
        """
        Set_USB_Data 控制器UART_USB接口波特率设置

        :param baudrate:波特率：9600,19200,38400,115200和460800，若用户设置其他数据，控制器会默认按照460800处理。
        :return:
        """

        self.pDll.Set_USB_Data.argtypes = (ctypes.c_int, ctypes.c_int)
        self.pDll.Set_USB_Data.restype = self.check_error

        tag = self.pDll.Set_USB_Data(self.nSocket, baudrate)

        logger_.info(f'Set_USB_Data:{tag}')

        return tag

    def Set_RS485(self, baudrate):
        """
        Set_RS485 控制器RS485接口波特率设置

        :param baudrate:波特率：9600,19200,38400,115200和460800，若用户设置其他数据，控制器会默认按照460800处理。
        :return:
        """

        self.pDll.Set_RS485.argtypes = (ctypes.c_int, ctypes.c_int)
        self.pDll.Set_RS485.restype = self.check_error

        tag = self.pDll.Set_RS485(self.nSocket, baudrate)

        logger_.info(f'Set_RS485:{tag}')

        return tag

    def Set_Arm_Power(self, cmd, block=True):
        """
        Set_Arm_Power 设置机械臂电源
        param cmd true-上电，   false-断电
        param block RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:
        """

        self.pDll.Set_Arm_Power.argtypes = (
            ctypes.c_int, ctypes.c_bool, ctypes.c_bool)
        self.pDll.Set_Arm_Power.restype = self.check_error

        tag = self.pDll.Set_Arm_Power(self.nSocket, cmd, block)

        logger_.info(f'Set_Arm_Power:{tag}')

        return tag

    def Get_Arm_Power_State(self, retry=0):
        """
        Get_Arm_Power_State      读取机械臂电源状态
        :return:
         power：获取到的机械臂电源状态：1-上电，0-断电
        """

        self.pDll.Get_Arm_Power_State.argtypes = (
            ctypes.c_int, ctypes.POINTER(ctypes.c_int))
        self.pDll.Get_Arm_Power_State.restype = self.check_error

        power = ctypes.c_int()

        tag = self.pDll.Get_Arm_Power_State(self.nSocket, ctypes.byref(power))

        while tag and retry:
            logger_.info(f'Get_Arm_Power_State:{tag},retry is :{6 - retry}')
            tag = self.pDll.Get_Arm_Power_State(
                self.nSocket, ctypes.byref(power))

            retry -= 1

        return tag, power.value

    def Get_Arm_Software_Version(self, retry=0):
        """
        Get_Arm_Software_Version     读取软件版本号
        :return:读取到的用户接口内核版本号，实时内核版本号，实时内核子核心1版本号，实时内核子核心2版本号，机械臂型号，仅I系列机械臂支持[-I]

        """

        self.pDll.Get_Arm_Software_Version.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                                                       ctypes.c_char_p, ctypes.c_char_p]
        self.pDll.Get_Arm_Software_Version.restype = self.check_error

        # 创建字符串变量
        plan_version = ctypes.create_string_buffer(256)
        ctrl_version = ctypes.create_string_buffer(256)
        kernal1 = ctypes.create_string_buffer(256)
        kernal2 = ctypes.create_string_buffer(256)
        product_version = ctypes.create_string_buffer(
            256)  # or None if not needed

        # 调用 Get_Arm_Software_Version 函数
        tag = self.pDll.Get_Arm_Software_Version(self.nSocket, plan_version, ctrl_version, kernal1, kernal2,
                                                 product_version)

        while tag and retry:
            logger_.info(
                f'Get_Arm_Software_Version:{tag},retry is :{6 - retry}')
            tag = self.pDll.Get_Arm_Software_Version(self.nSocket, plan_version, ctrl_version, kernal1, kernal2,
                                                     product_version)

            retry -= 1

        return tag, plan_version.value.decode(), ctrl_version.value.decode(), kernal1.value.decode(), kernal2.value.decode(), product_version.value.decode()

    def Get_System_Runtime(self, retry=0):
        """
        Get_System_Runtime           读取控制器的累计运行时间
        :param retry:
        :return:读取结果,读取到的时间day,读取到的时间hour,读取到的时间min,读取到的时间sec
        """

        self.pDll.Get_System_Runtime.argtypes = (
            ctypes.c_int, ctypes.POINTER(
                ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
        self.pDll.Get_System_Runtime.restype = self.check_error

        day = ctypes.c_int()
        hour = ctypes.c_int()
        min = ctypes.c_int()
        sec = ctypes.c_int()

        tag = self.pDll.Get_System_Runtime(self.nSocket, ctypes.byref(day), ctypes.byref(hour),
                                           ctypes.byref(min), ctypes.byref(sec))

        while tag and retry:
            logger_.info(f'Get_System_Runtime:{tag},retry is :{6 - retry}')
            tag = self.pDll.Get_System_Runtime(self.nSocket, ctypes.byref(day), ctypes.byref(hour),
                                               ctypes.byref(min), ctypes.byref(sec))

            retry -= 1

        return tag, day.value, hour.value, min.value, sec.value

    def Clear_System_Runtime(self, block=True):
        """
        Clear_System_Runtime         清零控制器的累计运行时间
        param block                        RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Clear_System_Runtime.argtypes = (ctypes.c_int, ctypes.c_bool)

        self.pDll.Clear_System_Runtime.restype = self.check_error

        tag = self.pDll.Clear_System_Runtime(self.nSocket, block)

        logger_.info(f'Clear_System_Runtime:{tag}')

        return tag

    def Get_Joint_Odom(self):
        """
        Get_Joint_Odom               读取关节的累计转动角度
        :param retry: 如果失败一共尝试读取五次
        :return:
        """
        if self.code == 6:

            self.pDll.Get_Joint_Odom.argtypes = (
                ctypes.c_int, ctypes.c_float * 6)
            self.pDll.Get_Joint_Odom.restype = self.check_error

            odom = (ctypes.c_float * 6)()

            tag = self.pDll.Get_Joint_Odom(self.nSocket, odom)

        else:
            self.pDll.Get_Joint_Odom.argtypes = (
                ctypes.c_int, ctypes.c_float * 7)
            self.pDll.Get_Joint_Odom.restype = self.check_error

            odom = (ctypes.c_float * 7)()

            tag = self.pDll.Get_Joint_Odom(self.nSocket, odom)

        logger_.info(f'Get_Joint_Odom 关节的累计转动角度:{list(odom)}')
        return tag, list(odom)

    def Clear_Joint_Odom(self, block=True):
        """
        Clear_Joint_Odom         清零关节的累计转动角度
        param block                        RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Clear_Joint_Odom.argtypes = (ctypes.c_int, ctypes.c_bool)

        self.pDll.Clear_Joint_Odom.restype = self.check_error

        tag = self.pDll.Clear_Joint_Odom(self.nSocket, block)

        logger_.info(f'Clear_Joint_Odom:{tag}')

        return tag

    def Set_High_Speed_Eth(self, num, block=True):
        """
        Set_High_Speed_Eth         设置高速网口
        :param num  0-关闭  1-开启
        param block   RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Set_High_Speed_Eth.argtypes = (
            ctypes.c_int, ctypes.c_byte, ctypes.c_bool)

        self.pDll.Set_High_Speed_Eth.restype = self.check_error

        tag = self.pDll.Set_High_Speed_Eth(self.nSocket, num, block)

        logger_.info(f'Set_High_Speed_Eth:{tag}')

        return tag

    def Set_High_Ethernet(self, ip, mask, gateway):
        """
        Set_High_Ethernet            设置高速网口网络配置[配置通讯内容]
        :param ip: 网络地址
        :param mask: 子网掩码
        :param gateway: 网关
        :return: 0-成功，失败返回:错误码, rm_define.h查询.

        """

        self.pDll.Set_High_Ethernet.argtypes = (
            ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p)
        self.pDll.Set_High_Ethernet.restype = self.check_error

        ip = ctypes.c_char_p(ip.encode('utf-8'))
        mask = ctypes.c_char_p(mask.encode('utf-8'))
        gateway = ctypes.c_char_p(gateway.encode('utf-8'))

        tag = self.pDll.Set_High_Ethernet(self.nSocket, ip, mask, gateway)

        logger_.info(f'Set_High_Ethernet:{tag}')

        return tag

    def Get_High_Ethernet(self, retry=0):
        """
        Get_High_Ethernet            获取高速网口网络配置[配置通讯内容]
        :param retry: 最大尝试次数
        :return: 成功返回 ip,mask,gateway,mac 否则None

        """

        self.pDll.Get_High_Ethernet.argtypes = (
            ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p)
        self.pDll.Get_High_Ethernet.restype = self.check_error

        ip = ctypes.create_string_buffer(255)
        mask = ctypes.create_string_buffer(255)
        gateway = ctypes.create_string_buffer(255)
        mac = ctypes.create_string_buffer(255)

        tag = self.pDll.Get_High_Ethernet(self.nSocket, ip, mask, gateway, mac)

        while tag and retry:
            logger_.info(f'Get_High_Ethernet:{tag},retry is :{6 - retry}')
            tag = self.pDll.Get_High_Ethernet(
                self.nSocket, ip, mask, gateway, mac)

            retry -= 1

        return tag, ip.value.decode(), mask.value.decode(), gateway.value.decode(), mac.value.decode()

    def Save_Device_Info_All(self):
        """

        Save_Device_Info_All 保存所有参数
        :return:0-成功，失败返回:错误码, rm_define.h查询.

        """

        tag = self.pDll.Save_Device_Info_All(self.nSocket)
        logger_.info(f'Save_Device_Info_All:{tag}')

        return tag

    def Set_NetIP(self, ip):
        """
        Set_NetIP                    配置有线网卡IP地址[-I]
        :param ip:网络地址
        :return:

        """

        self.pDll.Set_NetIP.argtypes = (ctypes.c_int, ctypes.c_char_p)

        ip = ctypes.c_char_p(ip.encode('utf-8'))

        tag = self.pDll.Set_NetIP(self.nSocket, ip)

        logger_.info(f'Set_NetIP:{tag}')

        return tag

    def Get_Wired_Net(self, retry=0):
        """
        Get_Wired_Net                查询有线网卡网络信息[-I]
        :param retry：接口调用失败后最多调用次数
        :return: ip，mask，gateway，mac
        """

        self.pDll.Get_Wired_Net.argtypes = (
            ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p)
        self.pDll.Get_Wired_Net.restype = self.check_error

        ip = ctypes.create_string_buffer(255)
        mask = ctypes.create_string_buffer(255)
        mac = ctypes.create_string_buffer(255)

        tag = self.pDll.Get_Wired_Net(self.nSocket, ip, mask, mac)

        while tag and retry:
            logger_.info(f'Get_Wired_Net:{tag},retry is :{6 - retry}')
            tag = self.pDll.Get_Wired_Net(self.nSocket, ip, mask, mac)

            retry -= 1

        return tag, ip.value.decode(), mask.value.decode(), mac.value.decode()

    def Get_Wifi_Net(self, retry=0):
        """
        Get_Wifi_Net         查询无线网卡网络信息[-I]
        :param retry：接口调用失败后最多调用次数
        :return:  wifi_net
        """

        self.pDll.Get_Wifi_Net.argtypes = (
            ctypes.c_int, ctypes.POINTER(WiFi_Info))
        self.pDll.Get_Wifi_Net.restype = self.check_error

        wifi_net = WiFi_Info()

        tag = self.pDll.Get_Wifi_Net(self.nSocket, wifi_net)

        while tag and retry:
            logger_.info(f'Get_Wifi_Net:{tag},retry is :{6 - retry}')
            tag = self.pDll.Get_Wifi_Net(self.nSocket, wifi_net)

            retry -= 1

        # if tag == 0:
        #     wifi_net = [wifi_net.ip, wifi_net.mac, wifi_net.mask, wifi_net.mode, wifi_net.password, wifi_net.ssid]

        return tag, wifi_net

    def Set_Net_Default(self):
        """
        Set_Net_Default 恢复网络出厂设置
        :return:
        """

        tag = self.pDll.Set_Net_Default(self.nSocket)

        logger_.info(f'Set_Net_Default:{tag}')

        return tag

    def Clear_System_Err(self, block=True):
        """
        Clear_System_Err 清除系统错误代码
        :param block: RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Clear_System_Err.argtypes = (ctypes.c_int, ctypes.c_bool)
        self.pDll.Clear_System_Err.restype = self.check_error

        tag = self.pDll.Clear_System_Err(self.nSocket, block)
        logger_.info(f'Clear_System_Err:{tag}')

        return tag

    def Get_Arm_Software_Info(self):
        """
        Get_Arm_Software_Info  读取机械臂软件信息[-I]
        :return:  software_info  机械臂软件信息
        """

        self.pDll.Get_Arm_Software_Info.argtypes = (
            ctypes.c_int, ctypes.POINTER(ArmSoftwareInfo))
        self.pDll.Get_Arm_Software_Info.restype = self.check_error

        software_info = ArmSoftwareInfo()

        tag = self.pDll.Get_Arm_Software_Info(self.nSocket, software_info)

        return tag, software_info


class Set_IO():

    def Set_IO_Mode(self, io_num, io_mode):
        """
        设置数字IO模式[-I]
        :param io_num: IO端口号，范围：1~2
        :param io_mode: 模式，0-输入状态，1-输出状态,2-输入开始功能复用模式，3-输入暂停功能复用模式，4-输入继续功能复用模式，5-输入急停功能复用模式
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Set_IO_Mode.argtypes = (
            ctypes.c_int, ctypes.c_byte, ctypes.c_byte)
        self.pDll.Set_IO_Mode.restype = self.check_error

        tag = self.pDll.Set_IO_Mode(self.nSocket, io_num, io_mode)

        logger_.info(f'Set_IO_Mode:{tag}')

        return tag

    def Set_DO_State(self, io_num, state, block=True):
        """
        设置数字IO输出
        :param io_num: 通道号，1~4
        :param state                        true-高，   false-低
        :param block                        0-非阻塞，发送后立即返回；1-阻塞，等待控制器返回设置成功指令
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """
        self.pDll.Set_DO_State.argtypes = (
            ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_bool)
        self.pDll.Set_DO_State.restype = self.check_error

        tag = self.pDll.Set_DO_State(self.nSocket, io_num, state, block)
        logger_.info(f'Set_DO_State执行的结果：{tag}')

        return tag

    def Get_IO_State(self, num):
        """
        Get_IO_State 获取IO状态
        :param num 通道号，1~4
        :return: state,mode
        """

        self.pDll.Get_IO_State.argtypes = (
            ctypes.c_int, ctypes.c_byte, ctypes.POINTER(ctypes.c_byte), ctypes.POINTER(ctypes.c_byte))

        self.pDll.Get_IO_State.restype = self.check_error

        state = ctypes.c_byte()
        mode = ctypes.c_byte()
        tag = self.pDll.Get_IO_State(
            self.nSocket, num, ctypes.byref(state), ctypes.byref(mode))

        logger_.info(f'Get_IO_State:{tag}')

        return tag, state.value, mode.value

    def Get_DO_State(self, io_num):
        """
        Get_DO_State 查询数字IO输出状态（基础系列）
        :param io_num 通道号，1~4
        :return: state  mode指定数字IO通道返回的状态，1-高，   0-低
        """

        self.pDll.Get_DO_State.argtypes = (
            ctypes.c_int, ctypes.c_byte, ctypes.POINTER(ctypes.c_byte))

        self.pDll.Get_DO_State.restype = self.check_error

        state = ctypes.c_byte()
        tag = self.pDll.Get_DO_State(self.nSocket, io_num, ctypes.byref(state))

        logger_.info(f'Get_DO_State执行结果:{tag}')

        return tag, state.value

    def Get_DI_State(self, io_num):
        """
        Get_DI_State 查询数字IO输入状态（基础系列）
        :param io_num 通道号，1~3
        :return: state  mode指定数字IO通道返回的状态，1-高，   0-低
        """

        self.pDll.Get_DI_State.argtypes = (
            ctypes.c_int, ctypes.c_byte, ctypes.POINTER(ctypes.c_byte))

        self.pDll.Get_DI_State.restype = self.check_error

        state = ctypes.c_byte()
        tag = self.pDll.Get_DI_State(self.nSocket, io_num, ctypes.byref(state))

        logger_.info(f'Get_DI_State执行结果:{tag}')

        return tag, state.value

    def Set_AO_State(self, io_num, voltage, block=True):
        """
        设置模拟IO输出（基础系列）
        :param io_num: 通道号，1~4
        :param voltage: IO输出电压，分辨率0.001V，范围：0~10000，代表输出电压0v~10v
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Set_AO_State.argtypes = (
            ctypes.c_int, ctypes.c_byte, ctypes.c_float, ctypes.c_bool)
        self.pDll.Set_AO_State.restype = self.check_error

        tag = self.pDll.Set_AO_State(self.nSocket, io_num, voltage, block)

        logger_.info(f'Set_AO_State执行结果:{tag}')

        return tag

    def Get_AO_State(self, io_num):
        """
        Get_AO_State 查询数字IO输出状态（基础系列）
        :param io_num 通道号，1~4
        :return: voltage  IO输出电压，分辨率0.001V，范围：0~10000，代表输出电压0v~10v
        """

        self.pDll.Get_AO_State.argtypes = (
            ctypes.c_int, ctypes.c_byte, ctypes.POINTER(ctypes.c_byte))

        self.pDll.Get_AO_State.restype = self.check_error

        voltage = ctypes.c_byte()
        tag = self.pDll.Get_AO_State(
            self.nSocket, io_num, ctypes.byref(voltage))

        logger_.info(f'Get_AO_State执行结果:{tag}')

        return tag, voltage.value

    def Get_AI_State(self, io_num):
        """
        Get_AI_State 查询数字IO输入状态（基础系列）
        :param io_num 通道号，1~4
        :return: voltage  IO输出电压，分辨率0.001V，范围：0~10000，代表输出电压0v~10v
        """

        self.pDll.Get_AI_State.argtypes = (
            ctypes.c_int, ctypes.c_byte, ctypes.POINTER(ctypes.c_byte))

        self.pDll.Get_AI_State.restype = self.check_error

        voltage = ctypes.c_byte()
        tag = self.pDll.Get_AI_State(
            self.nSocket, io_num, ctypes.byref(voltage))

        logger_.info(f'Get_AI_State执行结果:{tag}')

        return tag, voltage.value

    def Get_IO_Input(self):
        """
        Get_IO_Input 查询所有数字和模拟IO的输入状态
        :return:
        """

        self.pDll.Get_IO_Input.argtypes = (
            ctypes.c_int, ctypes.c_int * 4, ctypes.c_float * 4)
        self.pDll.Get_IO_Input.restype = self.check_error

        DI_state = (ctypes.c_int * 4)()
        AI_voltage = (ctypes.c_float * 4)()

        tag = self.pDll.Get_IO_Input(self.nSocket, DI_state, AI_voltage)

        logger_.info(f'Get_IO_Input:{tag}')

        return tag, list(DI_state), list(AI_voltage)

    def Get_IO_Output(self):
        """
        Get_IO_Output 查询所有数字和模拟IO的输出状态
        :return:
        """

        self.pDll.Get_IO_Output.argtypes = (
            ctypes.c_int, ctypes.c_int * 4, ctypes.c_float * 4)
        self.pDll.Get_IO_Output.restype = self.check_error

        DO_state = (ctypes.c_int * 4)()
        AO_voltage = (ctypes.c_float * 4)()

        tag = self.pDll.Get_IO_Output(self.nSocket, DO_state, AO_voltage)

        logger_.info(f'Get_IO_Output:{tag}')

        return tag, list(DO_state), list(AO_voltage)

    def Set_Voltage(self, voltage_type):
        """
        Set_Voltage                  设置电源输出[-I]
        :param voltage_type: 电源输出类型，范围：0~3(0-0V，2-12V，3-24V)
        :return:
        """

        self.pDll.Set_Voltage.argtypes = (ctypes.c_int, ctypes.c_byte)

        self.pDll.Set_Voltage.restype = self.check_error

        tag = self.pDll.Set_Voltage(self.nSocket, voltage_type)

        logger_.info(f'Set_Voltage:{tag}')

        return tag

    def Get_Voltage(self):
        """
        Get_Voltage                  获取电源输出类型[-I]
        :return:电源输出类型，范围：0~3(0-0V，2-12V，3-24V)
        """

        self.pDll.Get_Voltage.argtypes = (
            ctypes.c_int, ctypes.POINTER(ctypes.c_byte))
        self.pDll.Get_Voltage.restype = self.check_error

        voltage_type = ctypes.c_byte()

        tag = self.pDll.Get_Voltage(self.nSocket, ctypes.byref(voltage_type))

        logger_.info(f'Get_Voltage:{tag}')

        return voltage_type.value


class Set_Tool_IO():

    def Set_Tool_DO_State(self, num, state, block=True):
        """
        Set_Tool_DO_State 设置工具端数字IO输出
        :param num: 通道号，1~2
        :param state: true-高，   false-低
        :param block: RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:

        """

        self.pDll.Set_Tool_DO_State.argtypes = (
            ctypes.c_int, ctypes.c_byte, ctypes.c_bool, ctypes.c_bool)
        self.pDll.Set_Tool_DO_State.restypes = ctypes.c_int

        tag = self.pDll.Set_Tool_DO_State(self.nSocket, num, state, block)

        logger_.info(f'Set_Tool_DO_State:{tag}')

        return tag

    def Set_Tool_IO_Mode(self, num, state, block=True):
        """
        Set_Tool_IO_Mode 设置数字IO模式输入
        :param num: 通道号，1~2
        :param state: 0输入，   1输出
        :param block: RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Set_Tool_IO_Mode.argtypes = (
            ctypes.c_int, ctypes.c_byte, ctypes.c_bool, ctypes.c_bool)
        self.pDll.Set_Tool_IO_Mode.restype = self.check_error

        tag = self.pDll.Set_Tool_IO_Mode(self.nSocket, num, state, block)

        logger_.info(f'Set_Tool_IO_Mode:{tag}')

        return tag

    def Get_Tool_IO_State(self):
        """
        Get_Tool_IO_State 获取数字IO状态
        :param io_mode: 0-输入模式，1-输出模式
        :param io_state: 0-低，1-高
        :return: io_mode,io_state
        """

        self.pDll.Get_Tool_IO_State.argtypes = (
            ctypes.c_int, ctypes.c_float * 2, ctypes.c_float * 2)
        self.pDll.Get_Tool_IO_State.restype = self.check_error

        io_mode = (ctypes.c_float * 2)()
        io_state = (ctypes.c_float * 2)()

        tag = self.pDll.Get_Tool_IO_State(self.nSocket, io_mode, io_state)

        return tag, list(io_mode), list(io_state)

    def Set_Tool_Voltage(self, type, block=True):
        """
        打开夹抓 设置工具端电压输出
        param ArmSocket socket句柄
        type 电源输出类型，0-0V，1-5V，2-12V，3-24V
        block True 阻塞 False 非阻塞
        return 0-成功，失败返回:错误码
        :return:
        """
        self.pDll.Set_Tool_Voltage.argtypes = (
            ctypes.c_int, ctypes.c_int, ctypes.c_bool)
        self.pDll.Set_Tool_Voltage.restype = self.check_error

        tag = self.pDll.Set_Tool_Voltage(self.nSocket, type, block)
        logger_.info(f'设置工作端电压输出结果：{tag}')
        return tag

    def Get_Tool_Voltage(self):
        """
        Get_Tool_Voltage 查询工具端电压输出
        :return:工具端电压输出
        """

        self.pDll.Get_Tool_Voltage.argtypes = (
            ctypes.c_int, ctypes.POINTER(ctypes.c_byte))

        voltage = ctypes.c_byte()

        tag = self.pDll.Get_Tool_Voltage(self.nSocket, ctypes.byref(voltage))

        logger_.info(f'Get_Tool_Voltage:{tag}')

        return tag, voltage.value


class Set_Gripper():
    def Set_Gripper_Pick(self, speed, force, block=True, timeout=30):
        """
        Set_Gripper_Pick_On 手爪力控夹取
        ArmSocket socket句柄
        speed 手爪夹取速度 ，范围 1~1000，无单位量纲 无
        force 力控阈值 ，范围 ：50~1000，无单位量纲 无
        block True 阻塞 False 非阻塞
        timeout 超时时间设置，阻塞模式生效，单位：秒
        return 0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Set_Gripper_Pick.argtypes = (
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_int)
        self.pDll.Set_Gripper_Pick.restype = self.check_error

        tag = self.pDll.Set_Gripper_Pick(
            self.nSocket, speed, force, block, timeout)
        logger_.info(f'Set_Gripper_Pick执行结果:{tag}')

        return tag

    def Set_Gripper_Release(self, speed, block=True, timeout=30):
        """
        Set_Gripper_Release 手爪松开
        ArmSocket socket句柄
        speed 手爪松开速度 ，范围 1~1000，无单位量纲
        block True 阻塞 False 非阻塞
        timeout 超时时间设置，阻塞模式生效，单位：秒
        return 0-成功，失败返回:错误码
-
        """

        self.pDll.Set_Gripper_Release.argtypes = (
            ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_int)
        self.pDll.Set_Gripper_Release.restype = self.check_error

        tag = self.pDll.Set_Gripper_Release(
            self.nSocket, speed, block, timeout)
        logger_.info(f'Set_Gripper_Release执行结果:{tag}')
        return tag

    def Set_Gripper_Route(self, min_limit, max_limit, block=True):
        """
        Set_Gripper_Route 设置手爪行程
        :param min_limit: 手爪最小开口，范围 ：0~1000，无单位量纲 无
        :param max_limit: 手爪最大开口，范围 ：0~1000，无单位量纲 无
        :param block: block RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return: 0-成功，失败返回:错误码, rm_define.h查询
        """

        self.pDll.Set_Gripper_Route.argtypes = (
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool)
        self.pDll.Set_Gripper_Route.restype = self.check_error

        tag = self.pDll.Set_Gripper_Route(
            self.nSocket, min_limit, max_limit, block)

        logger_.info(f'Set_Gripper_Route：{tag}')

        return tag

    def Set_Gripper_Pick_On(self, speed, force, block=True, timeout=30):
        """
        Set_Gripper_Pick_On 手爪力控持续夹取
        :param speed:手爪夹取速度 ，范围 1~1000，无单位量纲 无
        :param force:力控阈值 ，范围 ：50~1000，无单位量纲 无
        :param block:RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :param timeout:超时时间设置，阻塞模式生效，单位：秒
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Set_Gripper_Pick_On.argtypes = (
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_int)
        self.pDll.Set_Gripper_Pick_On.restype = self.check_error

        tag = self.pDll.Set_Gripper_Pick_On(
            self.nSocket, speed, force, block, timeout)

        logger_.info(f'Set_Gripper_Pick_On:{tag}')

        return tag

    def Set_Gripper_Position(self, position, block=True, timeout=30):
        """
        Set_Gripper_Position 设置手爪开口度
        :param position:手爪开口位置 ，范围 ：1~1000，无单位量纲 无
        :param block:RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :param timeout:超时时间设置，阻塞模式生效，单位：秒
        :return:0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Set_Gripper_Position.argtypes = (
            ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_int)
        self.pDll.Set_Gripper_Position.restype = self.check_error

        tag = self.pDll.Set_Gripper_Position(
            self.nSocket, position, block, timeout)

        logger_.info(f'Set_Gripper_Position:{tag}')

        return tag

    def Get_Gripper_State(self):
        """
        Get_Gripper_State 获取夹爪状态
        :return:gripper_state   夹爪状态
        """

        self.pDll.Get_Gripper_State.argtypes = (
            ctypes.c_int, ctypes.POINTER(GripperState))
        self.pDll.Get_Gripper_State.restype = self.check_error

        state = GripperState()
        tag = self.pDll.Get_Gripper_State(self.nSocket, ctypes.byref(state))
        logger_.info(f'Get_Gripper_State:{tag}')

        return tag, state


class Drag_Teach():
    def Start_Drag_Teach(self, block=True):
        """
        Start_Drag_Teach 开始控制机械臂进入拖动示教模式
        :param block:RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:0-成功，失败返回:错误码, rm_define.h查询.

        """

        self.pDll.Start_Drag_Teach.argtypes = [ctypes.c_int, ctypes.c_bool]
        self.pDll.Start_Drag_Teach.restype = self.check_error

        tag = self.pDll.Start_Drag_Teach(self.nSocket, block)

        logger_.info(f'Start_Drag_Teach:{tag}')

        return tag

    def Stop_Drag_Teach(self, block=True):
        """
        Stop_Drag_Teach 控制机械臂退出拖动示教模式
        :param block:RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:0-成功，失败返回:错误码, rm_define.h查询.

        """

        self.pDll.Stop_Drag_Teach.argtypes = [ctypes.c_int, ctypes.c_bool]
        self.pDll.Stop_Drag_Teach.restype = self.check_error

        tag = self.pDll.Stop_Drag_Teach(self.nSocket, block)

        logger_.info(f'Stop_Drag_Teach:{tag}')

        return tag

    def Run_Drag_Trajectory(self, block=True):
        """
        Run_Drag_Trajectory 控制机械臂复现拖动示教的轨迹，必须在拖动示教结束后才能使用，
                       同时保证机械臂位于拖动示教的起点位置。
                       若当前位置没有位于轨迹复现起点，请先调用Drag_Trajectory_Origin，否则会返回报错信息。
        :param block:RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:0-成功，失败返回:错误码, rm_define.h查询.

        """

        self.pDll.Run_Drag_Trajectory.argtypes = [ctypes.c_int, ctypes.c_bool]
        self.pDll.Run_Drag_Trajectory.restype = self.check_error

        tag = self.pDll.Run_Drag_Trajectory(self.nSocket, block)

        logger_.info(f'Run_Drag_Trajectory:{tag}')

        return tag

    def Pause_Drag_Trajectory(self, block=True):
        """
        Pause_Drag_Trajectory 控制机械臂在轨迹复现过程中的暂停
        :param block:RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:0-成功，失败返回:错误码, rm_define.h查询.

        """

        self.pDll.Pause_Drag_Trajectory.argtypes = [
            ctypes.c_int, ctypes.c_bool]
        self.pDll.Pause_Drag_Trajectory.restype = self.check_error

        tag = self.pDll.Pause_Drag_Trajectory(self.nSocket, block)

        logger_.info(f'Pause_Drag_Trajectory:{tag}')

        return tag

    def Continue_Drag_Trajectory(self, block=True):
        """
        Continue_Drag_Trajectory 控制机械臂在轨迹复现过程中暂停之后的继续，
                                   轨迹继续时，必须保证机械臂位于暂停时的位置，
                                  否则会报错，用户只能从开始位置重新复现轨迹。
        :param block:RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:0-成功，失败返回:错误码, rm_define.h查询.

        """

        self.pDll.Continue_Drag_Trajectory.argtypes = [
            ctypes.c_int, ctypes.c_bool]
        self.pDll.Continue_Drag_Trajectory.restype = self.check_error

        tag = self.pDll.Continue_Drag_Trajectory(self.nSocket, block)

        logger_.info(f'Continue_Drag_Trajectory:{tag}')

        return tag

    def Stop_Drag_Trajectory(self, block=True):
        """
        Stop_Drag_Trajectory 控制机械臂在轨迹复现过程中的停止
        :param block:RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:0-成功，失败返回:错误码, rm_define.h查询.

        """

        self.pDll.Stop_Drag_Trajectory.argtypes = [ctypes.c_int, ctypes.c_bool]
        self.pDll.Stop_Drag_Trajectory.restype = self.check_error

        tag = self.pDll.Stop_Drag_Trajectory(self.nSocket, block)

        logger_.info(f'Stop_Drag_Trajectory:{tag}')

        return tag

    def Drag_Trajectory_Origin(self, block=True):
        """
        Drag_Trajectory_Origin 轨迹复现前，必须控制机械臂运动到轨迹起点，
                               如果设置正确，机械臂将以20%的速度运动到轨迹起点
        :param block:RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:0-成功，失败返回:错误码, rm_define.h查询.

        """

        self.pDll.Drag_Trajectory_Origin.argtypes = [
            ctypes.c_int, ctypes.c_bool]
        self.pDll.Drag_Trajectory_Origin.restype = self.check_error

        tag = self.pDll.Drag_Trajectory_Origin(self.nSocket, block)

        logger_.info(f'Drag_Trajectory_Origin:{tag}')

        return tag

    def Start_Multi_Drag_Teach(self, mode, singular_wall, block=True):
        """
        Start_Multi_Drag_Teach       开始复合模式拖动示教
        :param mode: 拖动示教模式 0-电流环模式，1-使用末端六维力，只动位置，2-使用末端六维力 ，只动姿态， 3-使用末端六维力，位置和姿态同时动
        :param block:RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:
        """

        self.pDll.Start_Multi_Drag_Teach.argtypes = (
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool)
        self.pDll.Start_Multi_Drag_Teach.restype = self.check_error

        tag = self.pDll.Start_Multi_Drag_Teach(
            self.nSocket, mode, singular_wall, block)
        logger_.info(f'Start_Multi_Drag_Teach:{tag}')

        return tag

    def Set_Force_Postion(self, sensor, mode, direction, N, block=True):
        """
        Set_Force_Postion       力位混合控制
        :param sensor: 0-一维力；1-六维力
        :param mode: 0-基坐标系力控；1-工具坐标系力控；
        :param direction: 力控方向；0-沿X轴；1-沿Y轴；2-沿Z轴；3-沿RX姿态方向；4-沿RY姿态方向；5-沿RZ姿态方向
        :param N: 力的大小，单位N
        :param block: RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Set_Force_Postion.argtypes = (
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool)
        self.pDll.Set_Force_Postion.restype = self.check_error

        tag = self.pDll.Set_Force_Postion(
            self.nSocket, sensor, mode, direction, N, block)

        logger_.info(f'Set_Force_Postion:{tag}')

        return tag

    def Stop_Force_Postion(self, block=True):
        """
        Stop_Force_Postion 结束力位混合控制
        :param block:RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:0-成功，失败返回:错误码, rm_define.h查询.

        """

        self.pDll.Stop_Force_Postion.argtypes = [ctypes.c_int, ctypes.c_bool]
        self.pDll.Stop_Force_Postion.restype = self.check_error

        tag = self.pDll.Stop_Force_Postion(self.nSocket, block)

        logger_.info(f'Stop_Force_Postion:{tag}')

        return tag

    def Save_Trajectory(self, file_name):
        """
        Save_Trajectory              获取刚拖动过的轨迹，在拖动示教后调用
        :param filename:             轨迹要保存路径及名称，例: c:/rm_test.txt
        :return:                     0-成功，失败返回:错误码, rm_define.h查询.
        """
        self.pDll.Save_Trajectory.argtypes = (
            ctypes.c_int, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int))
        self.pDll.Save_Trajectory.restype = self.check_error
        file_name = ctypes.create_string_buffer(file_name.encode('utf-8'))
        num = ctypes.c_int()
        tag = self.pDll.Save_Trajectory(
            self.nSocket, file_name, ctypes.byref(num))
        time.sleep(1)
        logger_.info(f'Save_Trajectory:{tag}')

        return tag, num.value


class Six_Force():
    def Get_Force_Data(self):
        """
        Get_Force_Data 查询当前六维力传感器得到的力和力矩信息，若要周期获取力数据      周期不能小于50ms。
        :return:力和力矩信息
        """

        self.pDll.Get_Force_Data.argtypes = (
            ctypes.c_int, ctypes.c_float * 6, ctypes.c_float * 6, ctypes.c_float * 6, ctypes.c_float * 6)

        self.pDll.Get_Force_Data.restype = self.check_error

        force = (ctypes.c_float * 6)()
        zero_force = (ctypes.c_float * 6)()
        work_zero = (ctypes.c_float * 6)()
        tool_zero = (ctypes.c_float * 6)()

        tag = self.pDll.Get_Force_Data(
            self.nSocket, force, zero_force, work_zero, tool_zero)

        logger_.info(f'Get_Force_Data:{tag}')

        return tag, list(force), list(zero_force), list(work_zero), list(tool_zero)

    def Set_Force_Sensor(self):

        tag = self.pDll.Set_Force_Sensor(self.nSocket)
        logger_.info(f'Set_Force_Sensor:{tag}')

        return tag

    def Manual_Set_Force(self, type, joints):
        """
        Manual_Set_Force 手动设置六维力重心参数，六维力重新安装后，必须重新计算六维力所收到的初始力和重心。
        :param type: 点位；1~4，调用此函数四次
        :param joints: 关节角度
        :return:
        """

        if self.code == 6:
            self.pDll.Manual_Set_Force.argtypes = (
                ctypes.c_int, ctypes.c_int, ctypes.c_float * 6)
            self.pDll.Manual_Set_Force.restype = self.check_error

            joints = (ctypes.c_float * 6)(*joints)

            tag = self.pDll.Manual_Set_Force(self.nSocket, type, joints)

        else:
            self.pDll.Manual_Set_Force.argtypes = (
                ctypes.c_int, ctypes.c_int, ctypes.c_float * 7)
            self.pDll.Manual_Set_Force.restype = self.check_error

            joints = (ctypes.c_float * 7)(*joints)

            tag = self.pDll.Manual_Set_Force(self.nSocket, type, joints)

        logger_.info(f'Manual_Set_Force:{tag}')
        return tag

    def Stop_Set_Force_Sensor(self, block=True):
        """
        Stop_Set_Force_Sensor 在标定六/一维力过程中，如果发生意外，发送该指令，停止机械臂运动，退出标定流程
        :param block:RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:0-成功，失败返回:错误码, rm_define.h查询.

        """

        self.pDll.Stop_Set_Force_Sensor.argtypes = [
            ctypes.c_int, ctypes.c_bool]
        self.pDll.Stop_Set_Force_Sensor.restype = self.check_error

        tag = self.pDll.Stop_Set_Force_Sensor(self.nSocket, block)

        logger_.info(f'Stop_Set_Force_Sensor:{tag}')

        return tag

    def Clear_Force_Data(self, block=True):
        """
        Clear_Force_Data 将六维力数据清零，即后续获得的所有数据都是基于当前数据的偏移量
        :param block:RM_NONBLOCK-非阻塞，发送后立即返回; RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Clear_Force_Data.argtypes = (ctypes.c_int, ctypes.c_bool)
        self.pDll.Clear_Force_Data.restype = self.check_error

        tag = self.pDll.Clear_Force_Data(self.nSocket, block)

        logger_.info(f'Clear_Force_Data:{tag}')

        return tag


class Set_Hand():

    def Set_Hand_Seq(self, seq_num, block=1):
        """
        设置灵巧手目标动作序列
        """
        tag = self.pDll.Set_Hand_Seq(self.nSocket, seq_num, block)
        logger_.info(f'Set_Hand_Seq:{tag}')
        time.sleep(0.5)

        return tag

    def Set_Hand_Posture(self, posture_num, block=1):
        """
        设置灵巧手目标手势
        """
        tag = self.pDll.Set_Hand_Posture(self.nSocket, posture_num, block)
        logger_.info(f'Set_Hand_Posture:{tag}')
        time.sleep(1)

        return tag

    def Set_Hand_Angle(self, angle, block=True):
        """
        Set_Hand_Angle 设置灵巧手各关节角度
        :param angle:手指角度数组，6个元素分别代表6个自由度的角度。范围：0~1000.另外，-1代表该自由度不执行任何操作，保持当前状态
        :param block:RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:0-成功，失败返回:错误码, rm_define.h查询.

        """

        self.pDll.Set_Hand_Angle.argtypes = (
            ctypes.c_int, ctypes.c_int * 6, ctypes.c_bool)
        self.pDll.Set_Hand_Angle.restype = self.check_error

        angle = (ctypes.c_int * 6)(*angle)

        tag = self.pDll.Set_Hand_Angle(self.nSocket, angle, block)

        logger_.info(f'Set_Hand_Angle:{tag}')

        return tag

    def Set_Hand_Speed(self, speed, block=True):
        """
        Set_Hand_Speed 设置灵巧手各关节速度
        :param speed:灵巧手各关节速度设置，范围：1~1000
        :param block:RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Set_Hand_Speed.argtypes = (
            ctypes.c_int, ctypes.c_int, ctypes.c_bool)
        self.pDll.Set_Hand_Speed.restype = self.check_error

        tag = self.pDll.Set_Hand_Speed(self.nSocket, speed, block)

        logger_.info(f'Set_Hand_Speed:{tag}')

        return tag

    def Set_Hand_Force(self, force, block=True):
        """
        Set_Hand_Force 设置灵巧手各关节力阈值
        :param force 灵巧手各关节力阈值设置，范围：1~1000，代表各关节的力矩阈值（四指握力0~10N，拇指握力0~15N）。
        :param block:RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Set_Hand_Force.argtypes = (
            ctypes.c_int, ctypes.c_int, ctypes.c_bool)
        self.pDll.Set_Hand_Force.restype = self.check_error

        tag = self.pDll.Set_Hand_Force(self.nSocket, force, block)

        logger_.info(f'Set_Hand_Force:{tag}')

        return tag


class one_force():
    def Get_Fz(self):
        """
        Get_Fz 该函数用于查询末端一维力数据
        :return:末端一维力数据
        """

        self.pDll.Get_Fz.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                     ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float))
        self.pDll.Get_Fz.restype = self.check_error

        fz = ctypes.c_float()
        zero_fz = ctypes.c_float()
        work_fz = ctypes.c_float()
        tool_fz = ctypes.c_float()

        tag = self.pDll.Get_Fz(self.nSocket, ctypes.byref(fz), ctypes.byref(zero_fz), ctypes.byref(work_fz),
                               ctypes.byref(tool_fz))

        logger_.info(f'Get_Fz:{tag}')

        return tag, fz.value, zero_fz.value, work_fz.value, tool_fz.value

    def Clear_Fz(self, block=True):
        """
        Clear_Fz 该函数用于清零末端一维力数据
        :param block:RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Clear_Fz.argtypes = (ctypes.c_int, ctypes.c_bool)
        self.pDll.Clear_Fz.restype = self.check_error

        tag = self.pDll.Clear_Fz(self.nSocket, block)

        logger_.info(f'Clear_Fz:{tag}')

        return tag

    def Auto_Set_Fz(self):
        """
        Auto_Set_Fz 该函数用于自动一维力数据
        :return:0-成功，失败返回:错误码, rm_define.h查询.
        """

        tag = self.pDll.Auto_Set_Fz(self.nSocket)
        logger_.info(f'Auto_Set_Fz:{tag}')

        return tag

    def Manual_Set_Fz(self, joint1, joint2):
        """
        Manual_Set_Fz 该函数用于手动设置一维力数据
        :param joint1:
        :param joint2:
        :return:
        """

        le = self.code

        self.pDll.Manual_Set_Fz.argtypes = (
            ctypes.c_int, ctypes.c_float * le, ctypes.c_float * le)
        self.pDll.Manual_Set_Fz.restype = self.check_error

        joint1 = (ctypes.c_float * le)(*joint1)
        joint2 = (ctypes.c_float * le)(*joint2)

        tag = self.pDll.Manual_Set_Fz(self.nSocket, joint1, joint2)

        logger_.info(f'Manual_Set_Fz:{tag}')

        return tag


class ModbusRTU():
    def Set_Modbus_Mode(self, port, baudrate, timeout, block=True):
        """
        配置通讯端口 Modbus RTU 模式
        :param port:通讯端口，0-控制器RS485端口为RTU主站，1-末端接口板RS485接口为RTU从站，2-控制器RS485端口为RTU从站
        :param baudrate:波特率，支持 9600,115200,460800 三种常见波特率
        :param timeout:超时时间，单位百毫秒。
        :param block: RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Set_Modbus_Mode.argtypes = (
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool)
        self.pDll.Set_Modbus_Mode.restype = self.check_error

        tag = self.pDll.Set_Modbus_Mode(
            self.nSocket, port, baudrate, timeout, block)

        logger_.info(f'Set_Modbus_Mode:{tag}')

        return tag

    def Close_Modbus_Mode(self, port, block=True):
        """
        Close_Modbus_Mode 关闭通讯端口 Modbus RTU 模式

        :param port: 通讯端口，0-控制器RS485端口为RTU主站，1-末端接口板RS485接口为RTU从站，2-控制器RS485端口为RTU从站
        :param block: RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Close_Modbus_Mode.argtypes = (
            ctypes.c_int, ctypes.c_int, ctypes.c_bool)
        self.pDll.Close_Modbus_Mode.restypes = ctypes.c_int

        tag = self.pDll.Close_Modbus_Mode(self.nSocket, port, block)

        logger_.info(f'Close_Modbus_Mode:{tag}')

        return tag

    def Set_Modbustcp_Mode(self, ip, port, timeout):
        """
        Set_Modbustcp_Mode配置连接 ModbusTCP 从站--I系列
        :param ip: 从机IP地址
        :param port: 端口号
        :param timeout: 超时时间，单位秒。
        :return:0-成功，失败返回:错误码, rm_define.h查询.
        """
        self.pDll.Set_Modbustcp_Mode.argtypes = (
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_int)
        self.pDll.Set_Modbustcp_Mode.restype = self.check_error

        ip = ctypes.c_char_p(ip.encode('utf-8'))
        tag = self.pDll.Set_Modbustcp_Mode(self.nSocket, ip, port, timeout)

        logger_.info(f'Set_Modbustcp_Mode:{tag}')

        return tag

    def Close_Modbustcp_Mode(self):
        """
        Close_Modbustcp_Mode 配置关闭 ModbusTCP 从站--I系列
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Close_Modbustcp_Mode.argtype = ctypes.c_int
        self.pDll.Close_Modbustcp_Mode.restypes = ctypes.c_int

        tag = self.pDll.Close_Modbustcp_Mode(self.nSocket)

        logger_.info(f'Close_Modbustcp_Mode:{tag}')

        return tag

    def Get_Read_Coils(self, port, address, num, device):
        """
        Get_Read_Coils 读线圈
        :param port: 通讯端口，0-控制器 RS485 端口，1-末端接口板 RS485 接口，3-控制器 ModbusTCP 设备
        :param address: 线圈起始地址
        :param num:要读的线圈的数量，该指令最多一次性支持读 8 个线圈数据，即返回的数据不会一个字节
        :param device:外设设备地址
        :return:返回离散量
        """

        self.pDll.Get_Read_Coils.argtypes = (
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int))
        self.pDll.Get_Read_Coils.restype = self.check_error

        coils_data = ctypes.c_int()

        tag = self.pDll.Get_Read_Coils(
            self.nSocket, port, address, num, device, ctypes.byref(coils_data))

        return tag, coils_data.value

    def Get_Read_Input_Status(self, port, address, num, device):
        """
        Get_Read_Input_Status 读离散量输入
        :param port: 通讯端口，0-控制器 RS485 端口，1-末端接口板 RS485 接口，3-控制器 ModbusTCP 设备
        :param address: 线圈起始地址
        :param num:要读的线圈的数量，该指令最多一次性支持读 8 个线圈数据，即返回的数据不会一个字节
        :param device:外设设备地址
        :return:返回离散量
        """

        self.pDll.Get_Read_Input_Status.argtypes = (
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int))
        self.pDll.Get_Read_Input_Status.restype = self.check_error

        coils_data = ctypes.c_int()

        tag = self.pDll.Get_Read_Input_Status(
            self.nSocket, port, address, num, device, ctypes.byref(coils_data))

        return tag, coils_data.value

    def Get_Read_Holding_Registers(self, port, address, device):
        """
        Get_Read_Holding_Registers 读保持寄存器
        :param port: 通讯端口，0-控制器 RS485 端口，1-末端接口板 RS485 接口，3-控制器 ModbusTCP 设备
        :param address: 线圈起始地址
        :param device: 外设设备地址
        :return:返回离散量
        """

        self.pDll.Get_Read_Holding_Registers.argtypes = (
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int))

        self.pDll.Get_Read_Holding_Registers.restype = self.check_error

        coils_data = ctypes.c_int()

        tag = self.pDll.Get_Read_Holding_Registers(
            self.nSocket, port, address, device, ctypes.byref(coils_data))

        return tag, coils_data.value

    def Get_Read_Input_Registers(self, port, address, device):
        """
        Get_Read_Input_Registers 读输入寄存器
        :param port: 通讯端口，0-控制器 RS485 端口，1-末端接口板 RS485 接口，3-控制器 ModbusTCP 设备
        :param address: 线圈起始地址
        :param device: 外设设备地址
        :return:返回离散量
        """

        self.pDll.Get_Read_Input_Registers.argtypes = (
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int))

        self.pDll.Get_Read_Input_Registers.restype = self.check_error

        coils_data = ctypes.c_int()

        tag = self.pDll.Get_Read_Input_Registers(
            self.nSocket, port, address, device, ctypes.byref(coils_data))

        return tag, coils_data.value

    def Write_Single_Coil(self, port, address, data, device, block=True):
        """
        Write_Single_Coil 写单圈数据
        :param port: 通讯端口，0-控制器 RS485 端口，1-末端接口板 RS485 接口，3-控制器 ModbusTCP 设备
        :param address: 线圈起始地址
        :param data: 要读的线圈的数量，该指令最多一次性支持读 8 个线圈数据，即返回的数据不会一个字节
        :param device: 外设设备地址
        :param block: RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Write_Single_Coil.argtypes = (
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool)
        self.pDll.Write_Single_Coil.restype = self.check_error

        tag = self.pDll.Write_Single_Coil(
            self.nSocket, port, address, data, device, block)

        logger_.info(f'Write_Single_Coil:{tag}')

        return tag

    def Write_Coils(self, port, address, num, coils_data, device, block=True):
        """
        brief Write_Coils 写多圈数据
        param port: 通讯端口，0-控制器 RS485 端口，1-末端接口板 RS485 接口，3-控制器 ModbusTCP 设备
        param address: 线圈起始地址
        param num: 写线圈个数，每次写的数量不超过160个
        param coils_data: 要写入线圈的数据数组，类型：byte。若线圈个数不大于8，则写入的数据为1个字节；否则，则为多个数据的数组
        param device: 外设设备地址
        param block: RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        return
        """
        device_num = int(num // 8 + 1)
        self.pDll.Write_Coils.argtypes = (
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_byte *
            device_num, ctypes.c_int,
            ctypes.c_bool)
        self.pDll.Write_Coils.restype = self.check_error

        coils_data = (ctypes.c_byte * device_num)(*coils_data)

        tag = self.pDll.Write_Coils(
            self.nSocket, port, address, num, coils_data, device, block)

        logger_.info(f'Write_Coils:{tag}')

        return tag

    def Write_Single_Register(self, port, address, data, device, block=True):
        """
        Write_Single_Register 写单个寄存器
        :param port: 通讯端口，0-控制器 RS485 端口，1-末端接口板 RS485 接口，3-控制器 ModbusTCP 设备
        :param address: 线圈起始地址
        :param data: 要读的线圈的数量，该指令最多一次性支持读 8 个线圈数据，即返回的数据不会一个字节
        :param device: 外设设备地址
        :param block:  RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Write_Single_Register.argtypes = (
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool)
        self.pDll.Write_Single_Register.restype = self.check_error

        tag = self.pDll.Write_Single_Register(
            self.nSocket, port, address, data, device, block)

        logger_.info(f'Write_Single_Register:{tag}')

        return tag

    def Write_Registers(self, port, address, num, single_data, device, block=True):
        """
         Write_Registers 写多个寄存器
        :param port: 通讯端口，0-控制器 RS485 端口，1-末端接口板 RS485 接口，3-控制器 ModbusTCP 设备
        :param address: 寄存器起始地址
        :param num: 写寄存器个数，寄存器每次写的数量不超过10个
        :param single_data: 要写入寄存器的数据数组，类型：byte
        :param device: 外设设备地址
        :param block:  RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:0-成功，失败返回:错误码, rm_define.h查询.
        """
        single_data_num = int(num * 2)

        self.pDll.Write_Registers.argtypes = (
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_byte *
            single_data_num, ctypes.c_int,
            ctypes.c_bool)
        self.pDll.Write_Registers.restype = self.check_error

        single_data = (ctypes.c_byte * single_data_num)(*single_data)

        tag = self.pDll.Write_Registers(
            self.nSocket, port, address, num, single_data, device, block)

        logger_.info(f'Write_Registers:{tag}')

        return tag

    def Read_Multiple_Holding_Registers(self, port, address, num, device):
        """
        Read_Multiple_Holding_Registers  读多个保存寄存器
        :param port: 0-控制器 RS485 端口，1-末端接口板 RS485 接口，3-控制器 ModbusTCP 设备
        :param address: 寄存器起始地址
        :param num: 2<num<13要读的寄存器的数量，该指令最多一次性支持读12个寄存器数据，即24个byte
        :param device: 外设设备地址
        :return: coils_data(线圈状态)
        """
        le = int(num * 2)
        self.pDll.Read_Multiple_Holding_Registers.argtypes = (
            ctypes.c_int, ctypes.c_byte, ctypes.c_int, ctypes.c_byte, ctypes.c_int, ctypes.c_int * le)
        self.pDll.Read_Multiple_Holding_Registers.restype = self.check_error

        coils_data = (ctypes.c_int * le)()

        tag = self.pDll.Read_Multiple_Holding_Registers(
            self.nSocket, port, address, num, device, coils_data)

        return tag, list(coils_data)

    def Get_Read_Multiple_Coils(self, port, address, num, device):
        """
        Read_Multiple_Coils  读多圈数据
        :param port: 0-控制器 RS485 端口，1-末端接口板 RS485 接口，3-控制器 ModbusTCP 设备
        :param address: 线圈起始地址
        :param num: 8< num <= 120 要读的线圈的数量，该指令最多一次性支持读 120 个线圈数据， 即15个byte
        :param device: 外设设备地址
        :return: coils_data(返回离散量)
        """

        self.pDll.Get_Read_Multiple_Coils.argtypes = (
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int * num)
        self.pDll.Get_Read_Multiple_Coils.restype = self.check_error

        coils_data = (ctypes.c_int * num)()

        tag = self.pDll.Get_Read_Multiple_Coils(
            self.nSocket, port, address, num, device, coils_data)

        return tag, list(coils_data)

    def Read_Multiple_Input_Registers(self, port, address, num, device):
        """
        Read_Multiple_Input_Registers  读多个输入寄存器
        :param port: 0-控制器 RS485 端口，1-末端接口板 RS485 接口，3-控制器 ModbusTCP 设备
        :param address: 寄存器起始地址
        :param num: 2<num<13要读的寄存器的数量，该指令最多一次性支持读12个寄存器数据，即24个byte
        :param device: 外设设备地址
        :return: coils_data(线圈状态)
        """
        le = int(num * 2)
        self.pDll.Read_Multiple_Input_Registers.argtypes = (
            ctypes.c_int, ctypes.c_byte, ctypes.c_int, ctypes.c_byte, ctypes.c_int, ctypes.c_int * le)
        self.pDll.Read_Multiple_Input_Registers.restype = self.check_error

        coils_data = (ctypes.c_int * le)()

        tag = self.pDll.Read_Multiple_Input_Registers(
            self.nSocket, port, address, num, device, coils_data)

        return tag, list(coils_data)


class Set_Lift():
    def Set_Lift_Height(self, height, speed, block=True):
        """
        Set_Lift_Height 导轨移动
        ArmSocket socket句柄
        height  目标高度，单位mm，范围：0~2600
        speed  速度百分比，1~100
        block True 阻塞 False 非阻塞
        :return:0-成功，失败返回:错误码
        """
        self.pDll.Set_Lift_Height.argtypes = (
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool)
        self.pDll.Set_Lift_Height.restype = self.check_error

        tag = self.pDll.Set_Lift_Height(self.nSocket, height, speed, block)
        logger_.info(f'Set_Lift_Height执行结果：{tag}')

        time.sleep(1)

        return tag

    def Set_Lift_Speed(self, speed):
        """
        Set_Lift_Speed           速度开环控制
        :param speed: speed-速度百分比，-100 ~100
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Set_Lift_Speed.argtypes = (ctypes.c_int, ctypes.c_int)
        self.pDll.Set_Lift_Speed.restype = self.check_error

        tag = self.pDll.Set_Lift_Speed(self.nSocket, speed)

        logger_.info(f'Set_Lift_Speed:{tag}')

        return tag

    def Get_Lift_State(self, retry=0):
        """
        Get_Lift_State           获取升降机构状态
        :param retry: 最大尝试次数
        Height:当前升降机构高度，单位：mm，精度：1mm，范围：0~2300
        Current:当前升降驱动电流，单位：mA，精度：1mA
        Err_flag:升降驱动错误代码，错误代码类型参考关节错误代码
        :return: Height，Current，Err_flag

        """

        self.pDll.Get_Lift_State.argtypes = (
            ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
        self.pDll.Get_Lift_State.restype = self.check_error

        height = ctypes.c_int()
        current = ctypes.c_int()
        err_flag = ctypes.c_int()

        tag = self.pDll.Get_Lift_State(self.nSocket, ctypes.byref(height), ctypes.byref(current),
                                       ctypes.byref(err_flag))

        while tag and retry:
            logger_.info(f'Get_Lift_State:{tag},retry is :{6 - retry}')
            tag = self.pDll.Get_Lift_State(self.nSocket, ctypes.byref(height), ctypes.byref(current),
                                           ctypes.byref(err_flag))

        return tag, height.value, current.value, err_flag.value

    # def Set_Arm_Dynamic_Parm(self, parm, block):
    #     """
    #     Set_Arm_Dynamic_Parm  重新设置机械臂动力学参数
    #     :param parm: 设置机械臂动力学参数
    #     :param block:RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
    #     :return:0-成功，失败返回:错误码, rm_define.h查询.
    #     """
    #     # RM_SERVICESHARED_EXPORT
    #     # int
    #     # Set_Arm_Dynamic_Parm(SOCKHANDLE
    #     # ArmSocket, const
    #     # float * parm, bool
    #     # block);


class Expand():
    def Expand_Set_Version(self, version):
        """
        Expand_Set_Version      扩展关节模式设置
        :param version:         0：表示关闭, 2：表示打开
        :return:                0-成功，失败返回:错误码, rm_define.h查询.
        """
        self.pDll.Expand_Set_Version.argtypes = (ctypes.c_int, ctypes.c_int)
        self.pDll.Expand_Set_Version.restype = self.check_error

        tag = self.pDll.Expand_Set_Version(self.nSocket, version)

        logger_.info(f'Expand_Set_Version:{tag}')

        return tag

    def Expand_Get_Version(self):
        """
        Expand_Get_Version      扩展关节模式设置
        :return:                0-成功，失败返回:错误码, rm_define.h查询.
        """
        self.pDll.Expand_Get_Version.argtypes = (
            ctypes.c_int, ctypes.POINTER(ctypes.c_int))
        self.pDll.Expand_Get_Version.restype = self.check_error

        ver = ctypes.c_int()

        tag = self.pDll.Expand_Get_Version(self.nSocket, ctypes.byref(ver))

        logger_.info(f'Expand_Get_Version:{tag}')

        return tag, ver.value

    def Expand_Get_State(self, retry=0):
        """
        Expand_Get_State           获取升降机构状态
        :param retry: 最大尝试次数
        pos:当前升降机构高度，单位：mm，精度：1mm，如果是旋转关节则为角度 单位度，精度0.001°
        Err_flag:升降驱动错误代码，错误代码类型参考关节错误代码
        Current:当前升降驱动电流，单位：mA，精度：1mA
        Mode:当前升降状态，0-空闲，1-正方向速度运动，2-正方向位置运动，3-负方向速度运动，4-负方向位置运动
        :return: pos，Current，Err_flag，Mode

        """

        self.pDll.Expand_Get_State.argtypes = (
            ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
        self.pDll.Get_Lift_State.restype = self.check_error

        pos = ctypes.c_int()
        mode = ctypes.c_int()
        current = ctypes.c_int()
        err_flag = ctypes.c_int()

        tag = self.pDll.Expand_Get_State(self.nSocket, ctypes.byref(pos), ctypes.byref(err_flag), ctypes.byref(current),
                                         ctypes.byref(mode))

        while tag and retry:
            logger_.info(f'Get_Lift_State:{tag},retry is :{6 - retry}')
            tag = self.pDll.Expand_Get_State(self.nSocket, ctypes.byref(pos), ctypes.byref(err_flag),
                                             ctypes.byref(current),
                                             ctypes.byref(mode))
        logger_.info(f'Expand_Get_State执行结果:{tag}')

        return tag, pos.value, err_flag.value, current.value, mode.value

    def Expand_Get_Config(self, retry=0):
        """
        Expand_Get_Config        获取电机参数
        ArmSocket                socket句柄
        config                   电机参数结构体
        :return:
        """

        self.pDll.Expand_Get_Config.argtypes = (
            ctypes.c_int, ctypes.POINTER(ExpandConfig))
        config = ExpandConfig()
        cp_ptr = ctypes.pointer(config)

        error_code = self.pDll.Expand_Get_Config(self.nSocket, cp_ptr)
        while error_code and retry:
            # sleep(0.3)
            logger_.warning(
                f"Failed to get expand config. Error Code: {error_code}\tRetry Count: {retry}")
            error_code = self.pDll.Expand_Get_Config(self.nSocket, cp_ptr)
            retry -= 1

        if error_code == 0:
            config = [config.rpm_max, config.rpm_acc,
                      config.conversin_coe, config.limit_max, config.limit_min]

        return error_code, config

    def Expand_Config(self, rpm_max, rpm_acc, conversin_coe):
        """
        Expand_Config           电机参数配置
        :param rpm_max:         关节的最大速度
        :param rpm_acc:         最大加速度
        :param conversin_coe:   减速比
        :return:                0-成功，失败返回:错误码, rm_define.h查询.
        """
        self.pDll.Expand_Config.argtypes = (
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
        self.pDll.Expand_Config.restype = self.check_error
        tag = self.pDll.Expand_Config(
            self.nSocket, rpm_max, rpm_acc, conversin_coe)

        logger_.info(f'Expand_Config:{tag}')

        return tag

    def Expand_Set_Pos(self, pos, speed, block=True):
        """
        Expand_Set_Pos 导轨移动
        ArmSocket socket句柄
        pos     升降关节精度 1mm   旋转关节精度 0.001°
        speed   50 表示最大速度的百分之五十,且速度必须大于0
        block   RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:0-成功，失败返回:错误码, rm_define.h查询.
        """
        self.pDll.Expand_Set_Pos.argtypes = (
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool)
        self.pDll.Expand_Set_Pos.restype = self.check_error

        tag = self.pDll.Expand_Set_Pos(self.nSocket, pos, speed, block)
        logger_.info(f'Expand_Set_Pos执行结果：{tag}')

        # time.sleep(1)

        return tag

    def Expand_Set_Speed(self, speed, block=0):
        """
        Expand_Set_Speed           速度开环控制
        :param speed: speed-速度百分比，-100 ~100
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Expand_Set_Speed.argtypes = (
            ctypes.c_int, ctypes.c_int, ctypes.c_int)
        self.pDll.Expand_Set_Speed.restype = self.check_error

        tag = self.pDll.Expand_Set_Speed(self.nSocket, speed, block)

        logger_.info(f'Expand_Set_Speed:{tag}')

        return tag


class UDP():
    def Get_Realtime_Push(self, retry=0):
        """
        Get_Realtime_Push        获取主动上报接口配置
        :param retry:
        :return:
        cycle                        获取广播周期，为5ms的倍数
        port                         获取广播的端口号
        enable                       获取使能，是否使能主动上上报
        error_code                   0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Get_Realtime_Push.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                                                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(
                                                    ctypes.c_bool),
                                                ctypes.POINTER(ctypes.c_int), ctypes.c_char_p)

        cycle = ctypes.c_int()
        port = ctypes.c_int()
        enable = ctypes.c_bool()
        force_coordinate = ctypes.c_int()
        ip = ctypes.create_string_buffer(255)
        error_code = self.pDll.Get_Realtime_Push(
            self.nSocket, cycle, port, enable, force_coordinate, ip)
        while error_code and retry:
            # sleep(0.3)
            logger_.warning(
                f"Failed to Get_Realtime_Push. Error Code: {error_code}\tRetry Count: {retry}")
            error_code = self.pDll.Get_Realtime_Push(
                self.nSocket, cycle, port, enable, force_coordinate, ip)
            retry -= 1

        return error_code, cycle.value, port.value, enable.value, force_coordinate.value, ip.value

    def Set_Realtime_Push(self, cycle=-1, port=-1, enable=True, force_coordinate=-1, ip=None):
        """
        Set_Realtime_Push            设置主动上报接口配置
        :param cycle:               设置广播周期，为5ms的倍数
        :param port:                设置广播的端口号
        :param enable:              设置使能，是否使能主动上上报
        :param force_coordinate:    系统外受力数据的坐标系，0为传感器坐标系 1为当前工作坐标系 2为当前工具坐标系
        :param ip:                  自定义的上报目标IP地址
        :return:                    0-成功，失败返回:错误码, rm_define.h查询.
        """
        self.pDll.Set_Realtime_Push.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool,
                                                ctypes.c_int, ctypes.c_char_p)
        self.pDll.Set_Realtime_Push.restype = self.check_error

        if ip is not None:
            ip = ctypes.c_char_p(ip.encode('utf-8'))
        else:
            ip = ctypes.c_char_p(b'')

        tag = self.pDll.Set_Realtime_Push(
            self.nSocket, cycle, port, enable, force_coordinate, ip)
        logger_.info(f'Set_Realtime_Push执行结果：{tag}')

        return tag

    def Realtime_Arm_Joint_State(self, RobotStatuscallback):
        """
        Realtime_Arm_Joint_State     机械臂状态主动上报
        :param RobotStatuscallback: 接收机械臂状态信息回调函数
        :return:
        """
        self.pDll.Realtime_Arm_Joint_State(RobotStatuscallback)


class Force_Position():
    def Start_Force_Position_Move(self, block=True):
        """
        Start_Force_Position_Move    开启透传力位混合控制补偿模式
        :param block: RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:0-成功，失败返回:错误码, rm_define.h查询.
        """

        tag = self.pDll.Start_Force_Position_Move(self.nSocket, block)

        logger_.info(f'Start_Force_Position_Move:{tag}')

        return tag

    def Force_Position_Move_Pose(self, pose, sensor, mode, dir, force, follow):
        """
        Force_Position_Move_Pose 透传力位混合补偿
        :param pose: 当前坐标系下目标位姿
        :param sensor: 所使用传感器类型，0-一维力，1-六维力
        :param mode: 模式，0-沿基坐标系，1-沿工具端坐标系
        :param dir: 力控方向，0~5分别代表X/Y/Z/Rx/Ry/Rz，其中一维力类型时默认方向为Z方向
        :param force: 力的大小 单位N
        :param follow:
        :return:
        """

        self.pDll.Force_Position_Move_Pose.argtypes = (ctypes.c_int, Pose, ctypes.c_byte, ctypes.c_byte,
                                                       ctypes.c_int, ctypes.c_float, ctypes.c_bool)
        self.pDll.Force_Position_Move_Pose.restype = self.check_error
        pose2 = Pose()

        pose2.position = Pos(*pose[:3])
        pose2.euler = Euler(*pose[3:])

        tag = self.pDll.Force_Position_Move_Pose(
            self.nSocket, pose2, sensor, mode, dir, force, follow)

        return tag

    def Force_Position_Move_Joint(self, joint, sensor, mode, dir, force, follow):
        """
        Force_Position_Move_Joint 透传力位混合补偿
        :param joint: 当前坐标系下目标joint
        :param sensor: 所使用传感器类型，0-一维力，1-六维力
        :param mode: 模式，0-沿基坐标系，1-沿工具端坐标系
        :param dir: 力控方向，0~5分别代表X/Y/Z/Rx/Ry/Rz，其中一维力类型时默认方向为Z方向
        :param force: 力的大小 单位N
        :param follow:
        :return:
        """

        le = self.code
        self.pDll.Force_Position_Move_Joint.argtypes = (ctypes.c_int, ctypes.c_float * le, ctypes.c_byte, ctypes.c_byte,
                                                        ctypes.c_int, ctypes.c_float, ctypes.c_bool)
        self.pDll.Force_Position_Move_Joint.restype = self.check_error
        joints = (ctypes.c_float * le)(*joint)

        tag = self.pDll.Force_Position_Move_Joint(
            self.nSocket, joints, sensor, mode, dir, force, follow)

        return tag

    def Stop_Force_Position_Move(self, block=True):
        """
        Stop_Force_Position_Move          停止透传力位混合控制补偿模式
        :param block: RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:
        """

        tag = self.pDll.Stop_Force_Position_Move(self.nSocket, block)

        logger_.info(f'Stop_Force_Position_Move:{tag}')

        return tag


class Algo:
    @classmethod
    def Algo_Init_Sys_Data(cls, dMode, bType):
        """
        :brief Algo_Init_Sys_Data           初始化算法依赖数据(不连接机械臂时调用， 连接机械臂会自动调用)。
        :param dMode                        机械臂型号，RobotType结构体
        :param bMode                        传感器型号，SensorType结构体
        """
        cls.pDll.Algo_Init_Sys_Data(dMode, bType)

    @classmethod
    def Algo_Set_Angle(cls, x, y, z):
        """
        :brief Algo_Set_Angle           设置安装角度
        :param x                        X轴安装角度 单位°
        :param y                        Y轴安装角度 单位°
        :param z                        z轴安装角度 单位°
        """
        cls.pDll.Algo_Set_Angle.argtypes = (
            ctypes.c_float, ctypes.c_float, ctypes.c_float)
        cls.pDll.Algo_Set_Angle(x, y, z)

    @classmethod
    def Algo_Get_Angle(cls):
        """
        :brief Algo_Get_Angle           获取安装角度
        :param x, y, z                   安装角度
        """

        cls.pDll.Algo_Get_Angle.argtypes = (
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float))
        x = ctypes.c_float()
        y = ctypes.c_float()
        z = ctypes.c_float()
        cls.pDll.Algo_Get_Angle(x, y, z)
        # matrix = Matrix()
        # cls.pDll.Algo_Get_Angle(ctypes.byref(matrix))

        return x.value, y.value, z.value

    @classmethod
    def Algo_Forward_Kinematics(cls, joint):
        """
        brief  Algo_Forward_Kinematics 正解函数
        param  joint                   关节1到关节6角度 单位°
        return Pose                    目标位姿
        """

        cls.pDll.Algo_Forward_Kinematics.restype = Pose
        joint = (ctypes.c_float * 7)(*joint)
        Pose_ = cls.pDll.Algo_Forward_Kinematics(joint)
        position = Pose_.position
        euler = Pose_.euler
        pose = [position.x, position.y, position.z,
                euler.rx, euler.ry, euler.rz]

        return pose

    @classmethod
    def Algo_Inverse_Kinematics(cls, q_in, q_pose, flag):
        """
        brief Algo_Inverse_Kinematics  逆解函数
        param q_in                     上一时刻关节角度 单位°
        param q_pose                   目标位姿
        param q_out                    输出的关节角度 单位°
        param flag                     姿态参数类别：0-四元数；1-欧拉角
        return                         SYS_NORMAL：计算正常，CALCULATION_FAILED：计算失败
        """
        q_out = (ctypes.c_float * 7)()
        q_in = (ctypes.c_float * 7)(*q_in)
        flag = ctypes.c_uint8(flag)
        po1 = Pose()
        po1.position = Pos(*q_pose[:3])
        if flag.value == 1:
            po1.euler = Euler(*q_pose[3:])
        else:
            po1.quaternion = Quat(*q_pose[3:])

        cls.pDll.Algo_Inverse_Kinematics.argtypes = (
            ctypes.c_float * 7, ctypes.POINTER(Pose), ctypes.c_float * 7, ctypes.c_uint8)
        tag = cls.pDll.Algo_Inverse_Kinematics(
            q_in, ctypes.byref(po1), q_out, flag)
        logger_.info(f'Algo_Inverse_Kinematics执行结果:{tag}')

        return tag, list(q_out)

    @classmethod
    def Algo_RotateMove(cls, curr_joint, rotate_axis, rotate_angle, choose_axis):
        """
        :brief  Algo_RotateMove         计算环绕运动位姿
        :param  curr_joint              当前关节角度 单位°
        :param  rotate_axis             旋转轴: 1:x轴, 2:y轴, 3:z轴
        :param  rotate_angle            旋转角度: 旋转角度, 单位(度)
        :param  choose_axis             指定计算时使用的坐标系
        :return Pose                    计算位姿结果
        """

        cls.pDll.Algo_RotateMove.restype = Pose
        cls.pDll.Algo_RotateMove.argtypes = (
            ctypes.c_float * 7, ctypes.c_int, ctypes.c_float, Pose)
        pose = Pose()

        pose.position = Pos(*choose_axis[:3])
        pose.euler = Euler(*choose_axis[3:])

        joint = (ctypes.c_float * 7)(*curr_joint)
        Pose_ = cls.pDll.Algo_RotateMove(
            joint, rotate_axis, rotate_angle, pose)

        position = Pose_.position
        euler = Pose_.euler
        return [position.x, position.y, position.z, euler.rx, euler.ry, euler.rz]

    @classmethod
    def Algo_End2Tool(cls, eu_end):
        """
        :brief  Algo_End2Tool           末端位姿转成工具位姿
        :param  eu_end                  基于世界坐标系和默认工具坐标系的末端位姿
        :return Pose                    基于工作坐标系和工具坐标系的末端位姿
        """

        cls.pDll.Algo_End2Tool.restype = Pose

        pose = Pose()

        pose.position = Pos(*eu_end[:3])
        pose.euler = Euler(*eu_end[3:])

        Pose_ = cls.pDll.Algo_End2Tool(pose)

        position = Pose_.position
        euler = Pose_.euler
        return [position.x, position.y, position.z, euler.rx, euler.ry, euler.rz]

    @classmethod
    def Algo_Tool2End(cls, eu_tool):
        """
        :brief  Algo_Tool2End           工具位姿转末端位姿
        :param  eu_tool                 基于工作坐标系和工具坐标系的末端位姿
        :return pose                    基于世界坐标系和默认工具坐标系的末端位姿
        """

        cls.pDll.Algo_Tool2End.restype = Pose

        pose = Pose()

        pose.position = Pos(*eu_tool[:3])
        pose.euler = Euler(*eu_tool[3:])

        Pose_ = cls.pDll.Algo_Tool2End(pose)

        position = Pose_.position
        euler = Pose_.euler
        return [position.x, position.y, position.z, euler.rx, euler.ry, euler.rz]

    @classmethod
    def Algo_Quaternion2Euler(cls, qua):
        """
        :brief  Algo_Quaternion2Euler   四元数转欧拉角
        :param  qua                     四元数
        :return Euler                   欧拉角
        """

        cls.pDll.Algo_Quaternion2Euler.restype = Euler

        qua = Quat(*qua)

        euler = cls.pDll.Algo_Quaternion2Euler(qua)

        return [euler.rx, euler.ry, euler.rz]

    @classmethod
    def Algo_Euler2Quaternion(cls, eu):
        """
        :param brief  Algo_Euler2Quaternion   欧拉角转四元数
        :param  eu                      欧拉角
        :param return Quat                    四元数
        """
        cls.pDll.Algo_Euler2Quaternion.restype = Quat

        eu = Euler(*eu)

        quat = cls.pDll.Algo_Euler2Quaternion(eu)
        return [quat.w, quat.x, quat.y, quat.z]

    @classmethod
    def Algo_Euler2Matrix(cls, eu):
        """
        :brief  Algo_Euler2Matrix       欧拉角转旋转矩阵
        :param  eu                      欧拉角
        :return Matrix                  旋转矩阵
        """

        cls.pDll.Algo_Euler2Matrix.restype = Matrix

        eu = Euler(*eu)

        matrix = cls.pDll.Algo_Euler2Matrix(eu)
        return matrix

    @classmethod
    def Algo_Pos2Matrix(cls, state):
        """
        :brief  Algo_Pos2Matrix         位姿转旋转矩阵
        :param  state                   位姿
        :return Matrix                  旋转矩阵
        """

        cls.pDll.Algo_Pos2Matrix.restype = Matrix

        pose = Pose()

        pose.position = Pos(*state[:3])
        pose.euler = Euler(*state[3:])

        matrix = cls.pDll.Algo_Pos2Matrix(pose)

        return matrix

    @classmethod
    def Algo_Matrix2Pos(cls, matrix):
        """
        :brief  Algo_Matrix2Pos         旋转矩阵转位姿
        :param  matrix                  旋转矩阵
        :return Pose                    位姿
        """
        cls.pDll.Algo_Matrix2Pos.restype = Pose
        Pose_ = cls.pDll.Algo_Matrix2Pos(matrix)

        return Pose_

    @classmethod
    def Algo_Base2WorkFrame(cls, matrix, state):
        """
        :brief  Algo_Base2WorkFrame     基坐标系转工作坐标系
        :param  matrix                  工作坐标系在基坐标系下的矩阵
        :param  state                   工具端坐标在基坐标系下位姿
        :return Pose                    基坐标系在工作坐标系下的位姿
        """

        # cls.pDll.Algo_Base2WorkFrame.argtypes = (Matrix,Pose)

        cls.pDll.Algo_Base2WorkFrame.restype = Pose

        pose = Pose()

        pose.position = Pos(*state[:3])
        pose.euler = Euler(*state[3:])

        Pose_ = cls.pDll.Algo_Base2WorkFrame(matrix, pose)

        return Pose_

    @classmethod
    def Algo_WorkFrame2Base(cls, matrix, state):
        """
        :brief  Algo_WorkFrame2Base     工作坐标系转基坐标系
        :param  matrix                  工作坐标系在基坐标系下的矩阵
        :param  state                   工具端坐标在工作坐标系下位姿
        :return Pose                    工作坐标系下的位姿
        """

        cls.pDll.Algo_WorkFrame2Base.restype = Pose
        pose = Pose()

        pose.position = Pos(*state[:3])
        pose.euler = Euler(*state[3:])

        Pose_ = cls.pDll.Algo_WorkFrame2Base(matrix, pose)

        return Pose_

    @classmethod
    def Algo_Cartesian_Tool(cls, curr_joint, move_lengthx, move_lengthy, move_lengthz):
        """
        brief  Algo_Cartesian_Tool     计算沿工具坐标系运动位姿
        param  curr_joint              当前关节角度
        param  move_lengthx            沿X轴移动长度，米为单位
        param  move_lengthy            沿Y轴移动长度，米为单位
        param  move_lengthz            沿Z轴移动长度，米为单位
        return Pose                    工作坐标系下的位姿
        """

        cls.pDll.Algo_Cartesian_Tool.argtypes = (
            ctypes.c_float * 7, ctypes.c_float, ctypes.c_float, ctypes.c_float)

        cls.pDll.Algo_Cartesian_Tool.restype = Pose

        curr_joint = (ctypes.c_float * 7)(*curr_joint)

        pose = cls.pDll.Algo_Cartesian_Tool(
            curr_joint, move_lengthx, move_lengthy, move_lengthz)

        return pose

    @classmethod
    def Algo_Set_WorkFrame(cls, frame):
        """
        brief  Algo_Set_WorkFrame      设置工作坐标系
        param  frame                    frame
        """

        cls.pDll.Algo_Set_WorkFrame.argtypes = [FRAME]

        cls.pDll.Algo_Set_WorkFrame(frame)

    @classmethod
    def Algo_Get_Curr_WorkFrame(cls):
        """
        :brief  Algo_Get_Curr_WorkFrame 获取当前工作坐标系
        :param  coord_work                  当前工作坐标系
        """

        coord_work = FRAME()

        cls.pDll.Algo_Get_Curr_WorkFrame(ctypes.byref(coord_work))

        return coord_work

    @classmethod
    def Algo_Set_ToolFrame(cls, coord_tool):
        """
        :brief Algo_Set_ToolFrame       设置工具坐标系
        :param frame                     坐标系信息
        """

        cls.pDll.Algo_Set_ToolFrame.argtypes = [FRAME]

        cls.pDll.Algo_Set_ToolFrame(coord_tool)

    @classmethod
    def Algo_Get_Curr_ToolFrame(cls):
        """
        :brief  Algo_Get_Curr_ToolFrame 获取当前工具坐标系
        :param  coord_tool                  当前工具坐标系
        """

        coord_tool = FRAME()

        cls.pDll.Algo_Get_Curr_ToolFrame(ctypes.byref(coord_tool))

        return coord_tool

    @classmethod
    def Algo_Set_Joint_Max_Limit(cls, joint_limit):
        """
        :brief Algo_Set_Joint_Max_Limit 设置关节最大限位
        :param joint_limit              单位°
        """
        joint_limit2 = (ctypes.c_float * 7)(*joint_limit)
        cls.pDll.Algo_Set_Joint_Max_Limit(joint_limit2)

    @classmethod
    def Algo_Get_Joint_Max_Limit(cls):
        """
        :brief Algo_Get_Joint_Max_Limit 获取关节最大限位
        :param joint_limit              返回关节最大限位
        """

        joint_limit3 = (ctypes.c_float * 7)()

        cls.pDll.Algo_Get_Joint_Max_Limit(joint_limit3)

        return list(joint_limit3)

    @classmethod
    def Algo_Set_Joint_Min_Limit(cls, joint_limit):
        """
        :brief Algo_Set_Joint_Min_Limit 设置关节最小限位
        :param joint_limit              单位°
        """
        joint_limit = (ctypes.c_float * 7)(*joint_limit)
        cls.pDll.Algo_Set_Joint_Min_Limit(joint_limit)

    @classmethod
    def Algo_Get_Joint_Min_Limit(cls):
        """
        :brief Algo_Get_Joint_Min_Limit 获取关节最小限位
        :param joint_limit              返回关节最小限位
        """

        joint_limit = (ctypes.c_float * 7)()

        cls.pDll.Algo_Get_Joint_Min_Limit(joint_limit)

        return list(joint_limit)

    @classmethod
    def Algo_Set_Joint_Max_Speed(cls, joint_slim_max):
        """
        :brief Algo_Set_Joint_Max_Speed 设置关节最大速度
        :param joint_slim_max           RPM
        """
        joint_slim_max = (ctypes.c_float * 7)(*joint_slim_max)
        cls.pDll.Algo_Set_Joint_Max_Speed(joint_slim_max)

    @classmethod
    def Algo_Get_Joint_Max_Speed(cls):
        """
        :brief Algo_Get_Joint_Max_Speed 获取关节最大速度
        :param joint_slim_max              返回关节最大速度
        """

        joint_slim_max = (ctypes.c_float * 7)()

        cls.pDll.Algo_Get_Joint_Max_Speed(joint_slim_max)

        return list(joint_slim_max)

    @classmethod
    def Algo_Set_Joint_Max_Acc(cls, joint_alim_max):
        """
        :brief Algo_Set_Joint_Max_Acc   设置关节最大加速度
        :param joint_alim_max           RPM/s
        """

        joint_alim_max = (ctypes.c_float * 6)(*joint_alim_max)
        cls.pDll.Algo_Set_Joint_Max_Acc(joint_alim_max)

    @classmethod
    def Algo_Get_Joint_Max_Acc(cls):
        """
        :brief Algo_Get_Joint_Max_Acc   获取关节最大加速度
        :param joint_alim_max           返回关节最大加速度
        """

        joint_alim_max = (ctypes.c_float * 6)()

        cls.pDll.Algo_Get_Joint_Max_Acc(ctypes.byref(joint_alim_max))

        return list(joint_alim_max)


class Online_programming():

    def Send_TrajectoryFile(self, file_name, plan_speed, auto_start, step_flag):
        """
        Send_TrajectoryFile          轨迹文件下发
        :param ArmSocket: socket句柄
        :param file_name: 轨迹文件完整路径 例: c:/rm_file.txt
        :param file_name_len: file_name 字段的长度
        :param plan_speed: 规划速度比例(0-100)
        :param auto_start: 设置默认在线编程文件 1-设置默认  0-设置非默认[-I]
        :param step_flag: 设置单步运行方式模式 1-设置单步模式  0-设置正常运动模式[-I]
        :return: err_line: 有问题的工程行数
        """

        self.pDll.Send_TrajectoryFile.argtypes = (
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_byte, ctypes.c_byte,
            ctypes.POINTER(ctypes.c_int))
        self.pDll.Send_TrajectoryFile.restype = ctypes.c_int

        file_name = ctypes.create_string_buffer(file_name.encode('utf-8'))
        err_line = ctypes.c_int()
        file_name_len = len(file_name)
        tag = self.pDll.Send_TrajectoryFile(self.nSocket, file_name, file_name_len, plan_speed, auto_start,
                                            step_flag, ctypes.byref(err_line))
        logger_.info(f'Send_TrajectoryFile: {tag}')

        return tag, err_line.value

    def Set_Plan_Speed(self, speed, block=True):
        """
        Set_Plan_Speed               轨迹规划中改变速度比例系数
        :param speed: 当前进度条的速度数据
        :param block: RM_NONBLOCK-非阻塞，发送后立即返回；RM_BLOCK-阻塞，等待控制器返回设置成功指令
        :return:0-成功，失败返回:错误码, rm_define.h查询.

        """

        self.pDll.Set_Plan_Speed.argtypes = (
            ctypes.c_int, ctypes.c_int, ctypes.c_bool)
        self.pDll.Set_Plan_Speed.restype = self.check_error

        tag = self.pDll.Set_Plan_Speed(self.nSocket, speed, block)

        logger_.info(f'Set_Plan_Speed:{tag}')

        return tag

    def Popup(self, content, block=True):
        """
        Popup 文件树弹窗提醒(本指令是控制器发送给示教器，返回值是示教器发送给控制器)
        :param content: 弹窗提示指令所在文件树的位置
        :return:0-成功，失败返回:错误码, rm_define.h查询.

        """

        self.pDll.Popup.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_bool)
        self.pDll.Popup.restype = self.check_error

        tag = self.pDll.Popup(self.nSocket, content, block)

        logger_.info(f'Popup:{tag}')

        return tag

    # def SetAngle(self, x, y, z):
    #     """
    #     setAngle  设置安装角度
    #     :param x: X轴安装角度 单位°
    #     :param y: Y轴安装角度 单位°
    #     :param z: Z轴安装角度 单位°
    #     :return:
    #     """
    #
    #     self.pDll.SetAngle.argtypes = (ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float)
    #     self.pDll.SetAngle.restype = self.check_error
    #
    #     tag = self.pDll.SetAngle(self.nSocket, x, y, z)
    #
    #     logger_.info(f'SetAngle:{tag}')
    #
    #     return tag


class Program_list():
    def Get_Program_Trajectory_List(self, page_num=0, page_size=0, vague_search=None):
        """
        Get_Program_Trajectory_List  查询在线编程程序列表
        :param programlist                  在线编程程序列表
                                    page_num:页码（0代表全部查询）
                                    page_size:每页大小（0代表全部查询）
                                    vague_search:模糊搜索 （传递此参数可进行模糊查询）
        """
        self.pDll.Get_Program_Trajectory_List.argtypes = [
            ctypes.c_int, ctypes.POINTER(ProgramTrajectoryData)]
        self.pDll.Get_Program_Trajectory_List.restype = self.check_error
        program_list = ProgramTrajectoryData()
        program_list.page_num = page_num
        program_list.page_size = page_size
        if vague_search is not None:
            program_list.vague_search = vague_search.encode('utf-8')
        else:
            program_list.vague_search = b''
        tag = self.pDll.Get_Program_Trajectory_List(
            self.nSocket, ctypes.byref(program_list))

        logger_.info(f'Get_Program_Trajectory_List:{tag}')
        return tag, program_list

    def Get_Program_Run_State(self):
        """
        Get_Program_Run_State：查询当前在线编程文件的运行状态
        :return: run_state:0未开始，1运行中，2暂停中
                id:运行轨迹编号，已存储轨迹的id，没有存储则为0，未运行则不返回
                plan_num：运行到的行数，未运行则不返回
                loop_num：存在循环指令的行数，未运行则不返回
                loop_cont：循环指令行数对应的运行次数，未运行则不返回
        """

        self.pDll.Get_Program_Run_State.argtypes = [
            ctypes.c_int, ctypes.POINTER(ProgramRunState)]
        self.pDll.Get_Program_Run_State.restype = self.check_error

        runState = ProgramRunState()

        result = self.pDll.Get_Program_Run_State(
            self.nSocket, ctypes.byref(runState))
        logger_.info(f'Get_Program_Run_State:{result}')

        return result, runState

    def Set_Program_ID_Start(self, id, speed=0, block=True):
        """
        Set_Program_ID_Start：运行指定编号在线编程
        :param id:运行指定的ID，1-100，存在轨迹可运行
        :param speed:1-100，需要运行轨迹的速度，可不提供速度比例，按照存储的速度运行
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """
        self.pDll.Set_Program_ID_Start.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool]
        self.pDll.Set_Program_ID_Start.restype = self.check_error

        result = self.pDll.Set_Program_ID_Start(self.nSocket, id, speed, block)
        logger_.info(f'Set_Program_ID_Start:{result}')

        return result

    def Delete_Program_Trajectory(self, id):
        """
        Delete_Program_Trajectory：可删除指定ID的轨迹
        :param id:指定需要删除的轨迹编号
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """
        self.pDll.Delete_Program_Trajectory.argtypes = [
            ctypes.c_int, ctypes.c_int]
        self.pDll.Delete_Program_Trajectory.restype = self.check_error

        result = self.pDll.Delete_Program_Trajectory(self.nSocket, id)
        logger_.info(f'Delete_Program_Trajectory:{result}')

        return result

    def Update_Program_Trajectory(self, id, plan_speed, project_name):
        """
        修改指定编号的轨迹信息
        :param id: 指定在线编程轨迹编号
        :param plan_speed: 更新后的规划速度比例
        :param project_name: 更新后的文件名称
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Update_Program_Trajectory.argtypes = (
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p)
        self.pDll.Update_Program_Trajectory.restype = self.check_error

        name = ctypes.c_char_p(project_name.encode('utf-8'))

        tag = self.pDll.Update_Program_Trajectory(
            self.nSocket, id, plan_speed, name)

        logger_.info(f'Update_Program_Trajectory:{tag}')

        return tag

    def Set_Default_Run_Program(self, id):
        """
        设置 IO 默认运行的在线编程文件编号
        :param id:设置 IO 默认运行的在线编程文件编号，支持 0-100，0 代表取消设置
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Set_Default_Run_Program.argtypes = (
            ctypes.c_int, ctypes.c_int)
        self.pDll.Set_Default_Run_Program.restype = self.check_error

        tag = self.pDll.Set_Default_Run_Program(self.nSocket, id)

        logger_.info(f'Set_Default_Run_Program:{tag}')

        return tag

    def Get_Default_Run_Program(self):
        """
        获取 IO 默认运行的在线编程文件编号
        :param
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        id:IO 默认运行的在线编程文件编号，支持 0-100，0 代表无默认
        """

        self.pDll.Get_Default_Run_Program.argtypes = (
            ctypes.c_int, ctypes.POINTER(ctypes.c_int))
        self.pDll.Get_Default_Run_Program.restype = self.check_error
        program_id = ctypes.c_int()
        tag = self.pDll.Get_Default_Run_Program(
            self.nSocket, ctypes.byref(program_id))

        logger_.info(f'Get_Default_Run_Program:{tag}')

        return tag, program_id.value


class Global_Waypoint():
    def Add_Global_Waypoint(self, waypoint):
        """
        新增全局路点
        :param waypoint: 新增全局路点参数结构体（无需输入新增全局路点时间）
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """
        self.pDll.Add_Global_Waypoint.argtypes = [
            ctypes.c_int, ctypes.POINTER(Waypoint)]
        self.pDll.Add_Global_Waypoint.restype = self.check_error

        tag = self.pDll.Add_Global_Waypoint(
            self.nSocket, ctypes.pointer(waypoint))

        logger_.info(f'Add_Global_Waypoint:{tag}')

        return tag

    def Update_Global_Waypoint(self, waypoint):
        """
        更新全局路点
        :param waypoint: 更新全局路点参数（无需输入新增全局路点时间）
        :return:
        """
        self.pDll.Update_Global_Waypoint.argtypes = [
            ctypes.c_int, ctypes.POINTER(Waypoint)]
        self.pDll.Update_Global_Waypoint.restype = self.check_error

        tag = self.pDll.Update_Global_Waypoint(
            self.nSocket, ctypes.pointer(waypoint))

        logger_.info(f'Update_Global_Waypoint:{tag}')

        return tag

    def Delete_Global_Waypoint(self, name):
        """
        删除全局路点
        :param name: 全局路点名称
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """
        self.pDll.Delete_Global_Waypoint.argtypes = (
            ctypes.c_int, ctypes.c_char_p)
        self.pDll.Delete_Global_Waypoint.restype = self.check_error

        name = ctypes.c_char_p(name.encode('utf-8'))

        tag = self.pDll.Delete_Global_Waypoint(self.nSocket, name)

        logger_.info(f'Delete_Global_Waypoint:{tag}')

        return tag

    def Get_Global_Point_List(self, page_num=0, page_size=0, vague_search=None):
        """
        Get_Global_Point_List  查询多个全局路点
        :param
                page_num:页码（0代表全部查询）
                page_size:每页大小（0代表全部查询）
                vague_search:模糊搜索 （传递此参数可进行模糊查询）
        :return:0-成功，失败返回:错误码, rm_define.h查询.
        """
        self.pDll.Get_Global_Point_List.argtypes = [
            ctypes.c_int, ctypes.POINTER(WaypointsList)]
        self.pDll.Get_Global_Point_List.restype = self.check_error
        point_list = WaypointsList()
        point_list.page_num = page_num
        point_list.page_size = page_size
        if vague_search is not None:
            point_list.vague_search = vague_search.encode('utf-8')
        else:
            point_list.vague_search = b''
        tag = self.pDll.Get_Global_Point_List(
            self.nSocket, ctypes.byref(point_list))

        logger_.info(f'Get_Global_Point_List:{tag}')
        return tag, point_list.to_output()

    def Given_Global_Waypoint(self, name):
        """
        Given_Global_Waypoint  查询指定全局路点
        :param name 指定全局路点名称
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """
        self.pDll.Given_Global_Waypoint.argtypes = [
            ctypes.c_int, ctypes.c_char_p, ctypes.POINTER(Waypoint)]
        self.pDll.Given_Global_Waypoint.restype = self.check_error
        point = Waypoint()
        name = ctypes.c_char_p(name.encode('utf-8'))
        tag = self.pDll.Given_Global_Waypoint(
            self.nSocket, name, ctypes.byref(point))

        logger_.info(f'Given_Global_Waypoint:{tag}')
        return tag, point.to_output()


class Electronic_Fencel():

    def Add_Electronic_Fence_Config(self, config):
        """
        :brief Add_Electronic_Fence_Config 新增几何模型参数
        :param config   几何模型参数
        :return 0-成功，失败返回:错误码, rm_define.h查询.
        """
        self.pDll.Add_Electronic_Fence_Config.argtypes = [
            ctypes.c_int, ElectronicFenceConfig]
        self.pDll.Add_Electronic_Fence_Config.restype = self.check_error

        result = self.pDll.Add_Electronic_Fence_Config(self.nSocket, config)
        logger_.info(f'Add_Electronic_Fence_Config:{result}')

        return result

    def Update_Electronic_Fence_Config(self, config):
        """
        :brief Add_Electronic_Fence_Config 更新几何模型参数
        :param config   几何模型参数
        :return 0-成功，失败返回:错误码, rm_define.h查询.
        """
        self.pDll.Update_Electronic_Fence_Config.argtypes = [
            ctypes.c_int, ElectronicFenceConfig]
        self.pDll.Update_Electronic_Fence_Config.restype = self.check_error

        result = self.pDll.Update_Electronic_Fence_Config(self.nSocket, config)
        logger_.info(f'Update_Electronic_Fence_Config:{result}')

        return result

    def Delete_Electronic_Fence_Config(self, name):
        """
        :brief Add_Electronic_Fence_Config 删除指定几何模型
        :param name   指定几何模型名称
        :return 0-成功，失败返回:错误码, rm_define.h查询.
        """
        self.pDll.Delete_Electronic_Fence_Config.argtypes = [
            ctypes.c_int, ctypes.c_char_p]
        self.pDll.Delete_Electronic_Fence_Config.restype = self.check_error

        name = ctypes.c_char_p(name.encode('utf-8'))
        result = self.pDll.Delete_Electronic_Fence_Config(self.nSocket, name)
        logger_.info(f'Delete_Electronic_Fence_Config:{result}')

        return result

    def Get_Electronic_Fence_List_Names(self):
        """
        :brief Get_Electronic_Fence_List_Names 查询所有几何模型名称
        :param ArmSocket socket句柄
        :param names   几何模型名称列表，长度为实际存在几何模型
        :param len  几何模型名称列表长度
        :return 0-成功，失败返回:错误码, rm_define.h查询.
        """
        self.pDll.Get_Electronic_Fence_List_Names.argtypes = [ctypes.c_int, ctypes.POINTER(ElectronicFenceNames),
                                                              ctypes.POINTER(ctypes.c_int)]
        self.pDll.Get_Electronic_Fence_List_Names.restype = self.check_error

        max_len = 10  # maximum number of tools
        # creates an array of FRAME_NAME
        names = (ElectronicFenceNames * max_len)()
        names_ptr = ctypes.POINTER(ElectronicFenceNames)(names)  #
        len_ = ctypes.c_int()

        result = self.pDll.Get_Electronic_Fence_List_Names(
            self.nSocket, names_ptr, ctypes.byref(len_))
        logger_.info(f'Get_Electronic_Fence_List_Names:{result}')

        job_names = [names[i].name.decode('utf-8') for i in range(len_.value)]
        return result, job_names, len_.value

    def Given_Electronic_Fence_Config(self, name):
        """
        :brief Given_Electronic_Fence_Config 查询指定几何模型参数
        :param ArmSocket socket句柄
        :param name   指定几何模型名称
        :param config  返回几何模型参数
        :return 0-成功，失败返回:错误码, rm_define.h查询.
        """
        self.pDll.Given_Electronic_Fence_Config.argtypes = [ctypes.c_int, ctypes.c_char_p,
                                                            ctypes.POINTER(ElectronicFenceConfig)]
        self.pDll.Given_Electronic_Fence_Config.restype = self.check_error

        name = ctypes.c_char_p(name.encode('utf-8'))
        config = ElectronicFenceConfig()

        result = self.pDll.Given_Electronic_Fence_Config(
            self.nSocket, name, ctypes.byref(config))
        logger_.info(f'Get_Electronic_Fence_List_Names:{result}')
        return result, config.to_output()

    def Get_Electronic_Fence_List_Info(self):
        """
        :brief Get_Electronic_Fence_List_Info 查询所有几何模型参数
        :param ArmSocket socket句柄
        :param config  几何模型信息列表，长度为实际存在几何模型
        :param len  几何模型信息列表长度
        :return 0-成功，失败返回:错误码, rm_define.h查询.
        """
        self.pDll.Get_Electronic_Fence_List_Info.argtypes = [ctypes.c_int, ctypes.POINTER(ElectronicFenceConfig),
                                                             ctypes.POINTER(ctypes.c_int)]
        self.pDll.Get_Electronic_Fence_List_Info.restype = self.check_error

        max_len = 10  # maximum number of tools
        # creates an array of FRAME_NAME
        config = (ElectronicFenceConfig * max_len)()
        config_ptr = ctypes.POINTER(ElectronicFenceConfig)(config)  #
        len_ = ctypes.c_int()

        result = self.pDll.Get_Electronic_Fence_List_Info(
            self.nSocket, config_ptr, ctypes.byref(len_))
        logger_.info(f'Get_Electronic_Fence_List_Info:{result}')

        return result, [config[i].to_output() for i in range(len_.value)], len_.value

    def Set_Electronic_Fence_Enable(self, enable, in_out_side, effective_region):
        """
        Set_Electronic_Fence_Enable          设置电子围栏使能状态
        :param enable: true代表使能，false代表禁使能
        :param in_out_side：0-机器人在电子围栏内部，1-机器人在电子围栏外部
        :param effective_region：0-针对整臂区域生效
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """
        self.pDll.Set_Electronic_Fence_Enable.argtypes = [
            ctypes.c_int, ctypes.c_bool, ctypes.c_int, ctypes.c_int]
        self.pDll.Set_Electronic_Fence_Enable.restype = self.check_error

        result = self.pDll.Set_Electronic_Fence_Enable(
            self.nSocket, enable, in_out_side, effective_region)
        logger_.info(f'Set_Electronic_Fence_Enable:{result}')

        return result

    def Get_Electronic_Fence_Enable(self):
        """
        Get_Electronic_Fence_Enable          获取电子围栏使能状态
        :param enable: true代表使能，false代表禁使能
        :param in_out_side：0-机器人在电子围栏内部，1-机器人在电子围栏外部
        :param effective_region：0-针对整臂区域生效
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """
        self.pDll.Get_Electronic_Fence_Enable.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_bool),
                                                          ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        self.pDll.Get_Electronic_Fence_Enable.restype = self.check_error

        enable = ctypes.c_bool()
        in_out_side = ctypes.c_int()
        effective_region = ctypes.c_int()
        result = self.pDll.Get_Electronic_Fence_Enable(self.nSocket, ctypes.byref(enable), ctypes.byref(in_out_side),
                                                       ctypes.byref(effective_region))
        logger_.info(f'Get_Electronic_Fence_Enable:{result}')

        return result, enable.value, in_out_side.value, effective_region.value

    def Set_Electronic_Fence_Config(self, config):
        """
        :brief Set_Electronic_Fence_Config 设置当前电子围栏参数
        :param config   电子围栏参数
        :return 0-成功，失败返回:错误码, rm_define.h查询.
        """
        # config = ElectronicFenceConfig()
        self.pDll.Set_Electronic_Fence_Config.argtypes = [
            ctypes.c_int, ElectronicFenceConfig]
        self.pDll.Set_Electronic_Fence_Config.restype = self.check_error

        result = self.pDll.Set_Electronic_Fence_Config(self.nSocket, config)
        logger_.info(f'Set_Electronic_Fence_Config:{result}')

        return result

    def Get_Electronic_Fence_Config(self):
        """
        :brief Get_Electronic_Fence_Config 获取当前电子围栏参数
        :param config   电子围栏参数
        :return 0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Get_Electronic_Fence_Config.argtypes = [
            ctypes.c_int, ctypes.POINTER(ElectronicFenceConfig)]
        self.pDll.Get_Electronic_Fence_Config.restype = self.check_error
        config = ElectronicFenceConfig()
        result = self.pDll.Get_Electronic_Fence_Config(
            self.nSocket, ctypes.byref(config))
        logger_.info(f'Get_Electronic_Fence_Config:{result}')

        return result, config.to_output()

    def Set_Virtual_Wall_Enable(self, enable, in_out_side, effective_region):
        """
        Set_Virtual_Wall_Enable          设置虚拟墙使能状态
        :param enable: true代表使能，false代表禁使能
        :param in_out_side：0-机器人在虚拟墙内部
        :param effective_region：1-针对末端生效
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """
        self.pDll.Set_Virtual_Wall_Enable.argtypes = [
            ctypes.c_int, ctypes.c_bool, ctypes.c_int, ctypes.c_int]
        self.pDll.Set_Virtual_Wall_Enable.restype = self.check_error

        result = self.pDll.Set_Virtual_Wall_Enable(
            self.nSocket, enable, in_out_side, effective_region)
        logger_.info(f'Set_Virtual_Wall_Enable:{result}')

        return result

    def Get_Virtual_Wall_Enable(self):
        """
        Get_Virtual_Wall_Enable          获取虚拟墙使能状态
        :param enable: true代表使能，false代表禁使能
        :param in_out_side：0-机器人在虚拟墙内部
        :param effective_region：1-针对末端生效
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """
        self.pDll.Get_Virtual_Wall_Enable.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_bool),
                                                      ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        self.pDll.Get_Virtual_Wall_Enable.restype = self.check_error

        enable = ctypes.c_bool()
        in_out_side = ctypes.c_int()
        effective_region = ctypes.c_int()
        result = self.pDll.Get_Virtual_Wall_Enable(self.nSocket, ctypes.byref(enable), ctypes.byref(in_out_side),
                                                   ctypes.byref(effective_region))
        logger_.info(f'Get_Virtual_Wall_Enable:{result}')

        return result, enable.value, in_out_side.value, effective_region.value

    def Set_Virtual_Wall_Config(self, config):
        """
        Set_Virtual_Wall_Config 设置当前虚拟墙参数
        :param config   虚拟墙参数
        :return 0-成功，失败返回:错误码, rm_define.h查询.
        """
        # config = ElectronicFenceConfig()
        self.pDll.Set_Virtual_Wall_Config.argtypes = [
            ctypes.c_int, ElectronicFenceConfig]
        self.pDll.Set_Virtual_Wall_Config.restype = self.check_error

        result = self.pDll.Set_Virtual_Wall_Config(self.nSocket, config)
        logger_.info(f'Set_Virtual_Wall_Config:{result}')

        return result

    def Get_Virtual_Wall_Config(self):
        """
        Get_Virtual_Wall_Config 获取当前虚拟墙参数
        :param config   虚拟墙参数
        :return 0-成功，失败返回:错误码, rm_define.h查询.
        """

        self.pDll.Get_Virtual_Wall_Config.argtypes = [
            ctypes.c_int, ctypes.POINTER(ElectronicFenceConfig)]
        self.pDll.Get_Virtual_Wall_Config.restype = self.check_error
        config = ElectronicFenceConfig()
        result = self.pDll.Get_Virtual_Wall_Config(
            self.nSocket, ctypes.byref(config))
        logger_.info(f'Get_Virtual_Wall_Config:{result}')

        return result, config.to_output()

    def Set_Self_Collision_Enable(self, enable):
        """
        :brief Set_Self_Collision_Enable 设置自碰撞安全检测使能状态
        :param ArmSocket socket句柄
        :param enable true代表使能，false代表禁使能
        :return 0-成功，失败返回:错误码, rm_define.h查询.
        """
        self.pDll.Set_Self_Collision_Enable.argtypes = [
            ctypes.c_int, ctypes.c_bool]
        self.pDll.Set_Self_Collision_Enable.restype = self.check_error

        result = self.pDll.Set_Self_Collision_Enable(self.nSocket, enable)
        logger_.info(f'Set_Self_Collision_Enable:{result}')

        return result

    def Get_Self_Collision_Enable(self):
        """
        Get_Self_Collision_Enable          获取自碰撞安全检测使能状态
        :param enable: true代表使能，false代表禁使能
        :return: 0-成功，失败返回:错误码, rm_define.h查询.
        """
        self.pDll.Get_Self_Collision_Enable.argtypes = [
            ctypes.c_int, ctypes.POINTER(ctypes.c_bool)]
        self.pDll.Get_Self_Collision_Enable.restype = self.check_error

        enable = ctypes.c_bool()

        result = self.pDll.Get_Self_Collision_Enable(
            self.nSocket, ctypes.byref(enable))
        logger_.info(f'Get_Self_Collision_Enable:{result}')

        return result, enable.value


class Arm(Set_Joint, Get_Joint, Tcp_Config, Tool_Frame, Work_Frame, Arm_State, Initial_Pose, Move_Plan, Teaching,
          Set_controller, Set_IO, Set_Tool_IO, Set_Gripper, Drag_Teach, Six_Force, Set_Hand, one_force,
          ModbusRTU, Set_Lift, Force_Position, Algo, Online_programming, Expand, UDP, Program_list, Electronic_Fencel,
          Global_Waypoint):
    pDll = ctypes.cdll.LoadLibrary(dllPath)

    def __init__(self, dev_mode, ip, pCallback=None):
        # RM_Callback = ctypes.CFUNCTYPE(None, CallbackData)
        self.code = dev_mode
        while self.code >= 10:
            self.code //= 10

        if pCallback is None:
            self.pDll.RM_API_Init(dev_mode, 0)  # API初始化
        else:
            self.pDll.RM_API_Init(dev_mode, pCallback)  # API初始化

        logger_.info('开始进行机械臂API初始化完毕')

        self.API_Version()

        # 连接机械臂
        byteIP = bytes(ip, "gbk")
        self.nSocket = self.pDll.Arm_Socket_Start(byteIP, 8080, 200)  # 连接机械臂

        state = self.pDll.Arm_Socket_State(self.nSocket)  # 查询机械臂连接状态

        if state:
            logger_.info(f'连接机械臂连接失败:{self.nSocket}')

        else:
            logger_.info(f'连接机械臂成功，句柄为:{self.nSocket}')

    def Arm_Socket_State(self):
        """
        Arm_Socket_State        查询机械臂连接状态
        :return:0-成功，失败返回:错误码, rm_define.h查询.
        """
        state = self.pDll.Arm_Socket_State(self.nSocket)  # 查询机械臂连接状态

        if state == 0:
            return state
        else:
            return errro_message[state]

    def API_Version(self):
        """
        API_Version          查询API版本信息
        return                       API版本号
        """
        self.pDll.API_Version.restype = ctypes.c_char_p
        api_name = self.pDll.API_Version()
        logger_.info(f'API_Version:{api_name.decode()}')
        time.sleep(0.5)

        return api_name.decode()

    def RM_API_UnInit(self):
        """
        API反初始化 释放资源
        :return:
        """
        tag = self.pDll.RM_API_UnInit()
        logger_.info(f'API反初始化 释放资源')
        return tag

    def Set_Arm_Run_Mode(self, mode):
        """
        设置机械臂模式（仿真/真实）
        mode                         模式 0:仿真 1:真实
        """
        self.pDll.Set_Arm_Run_Mode.argtypes = [ctypes.c_int, ctypes.c_int]
        self.pDll.Set_Arm_Run_Mode.restype = self.check_error

        result = self.pDll.Set_Arm_Run_Mode(self.nSocket, mode)
        logger_.info(f'Set_Arm_Run_Mode:{result}')

        return result

    def Get_Arm_Run_Mode(self):
        """
        获取机械臂模式（仿真/真实）
        mode                         模式 0:仿真 1:真实
        """
        self.pDll.Get_Arm_Run_Mode.argtypes = [
            ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self.pDll.Get_Arm_Run_Mode.restype = self.check_error

        mode = ctypes.c_int()
        result = self.pDll.Get_Arm_Run_Mode(self.nSocket, ctypes.byref(mode))
        logger_.info(f'Get_Arm_Run_Mode:{result}')

        return result, mode.value

    def Arm_Socket_Close(self):
        """
        关闭与机械臂的Socket连接
        :return:
        """
        self.pDll.Arm_Socket_Close(self.nSocket)
        logger_.info(f'关闭与机械臂的Socket连接')

    @staticmethod
    def check_error(tag):

        if tag == 0:
            return tag
        else:
            return errro_message[tag]
