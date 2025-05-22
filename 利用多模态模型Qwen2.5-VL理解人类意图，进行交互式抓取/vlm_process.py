import cv2
import numpy as np
import torch
import requests
import json
import re
import base64
import textwrap
import queue
import time
import io
import sounddevice as sd
from scipy.io.wavfile import write
from pydub import AudioSegment
from ultralytics.models.sam import Predictor as SAMPredictor


# ----------------------- 基础工具函数 -----------------------

def encode_np_array(image_np):
    """将 numpy 图像数组（BGR）编码为 base64 字符串"""
    success, buffer = cv2.imencode('.jpg', image_np)
    if not success:
        raise ValueError("无法将图像数组编码为 JPEG")
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64


def plot_coordinates(image_input, coords):
    """根据坐标打印简单提示，但不显示图像"""
    if not coords:
        print("[提示] 未提取到坐标信息。")
        return
    if "bbox" in coords:
        print(f"[标注] 边界框坐标: {coords['bbox']}")
    elif "point" in coords:
        print(f"[标注] 中心点坐标: {coords['point']}")
    elif "x" in coords and "y" in coords:
        print(f"[标注] 中心点坐标: ({coords['x']}, {coords['y']})")


# ----------------------- 多模态模型调用（Qwen） -----------------------

def generate_robot_actions(user_command, image_input=None):
    """
    使用 base64 的方式将 numpy 图像和用户文本指令传给 Qwen 多模态模型，
    要求模型返回两部分：
      - 模型返回内容中，第一部分为自然语言响应（说明为何选择该物体），
      - 紧跟其后的部分为纯 JSON 对象，格式如下：

        {
          "name": "物体名称",
          "bbox": [左上角x, 左上角y, 右下角x, 右下角y]
        }

    返回一个 dict，包含 "response" 和 "coordinates"。
    参数 image_input 为 numpy 数组（BGR 格式）。
    """
    url = "http://192.168.1.6:1034/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    system_prompt = textwrap.dedent("""\
    你是一个精密机械臂视觉控制系统，具备先进的多模态感知能力。请严格按照以下步骤执行任务：

    【图像分析阶段】
    1. 分析输入图像，识别图像中所有可见物体，并记录每个物体的边界框（左上角点和右下角点）及其类别名称。

    【指令解析阶段】
    2. 根据用户的自然语言指令，从识别的物体中筛选出最匹配的目标物体。

    【响应生成阶段】
    3. 输出格式必须严格如下：
    - 自然语言响应（仅包含说明为何选择该物体的文字,可以俏皮可爱地回应用户的需求，但是请注意，回答中应该只包含被选中的物体），
    - 紧跟其后，从下一行开始返回 **标准 JSON 对象**，格式如下：

    {
      "name": "物体名称",
      "bbox": [左上角x, 左上角y, 右下角x, 右下角y]
    }

    【注意事项】
    - JSON 必须从下一行开始；
    - 自然语言响应与 JSON 之间无其他额外文本；
    - JSON 对象不能有注释、额外文本或解释；
    - 坐标 bbox 必须为整数；
    - 只允许使用 "bbox" 作为坐标格式。
    """)

    messages = [{"role": "system", "content": system_prompt}]
    user_content = []

    if image_input is not None:
        base64_img = encode_np_array(image_input)
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_img}"
            }
        })

    user_content.append({"type": "text", "text": user_command})
    messages.append({"role": "user", "content": user_content})

    data = {
        "model": "qwen7b",
        "messages": messages
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=15)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        print("原始响应：", content)

        # 使用正则表达式查找 JSON 部分
        match = re.search(r'(\{.*\})', content, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                coord = json.loads(json_str)
            except Exception as e:
                print(f"[警告] JSON 解析失败：{e}")
                coord = {}
            natural_response = content[:match.start()].strip()
        else:
            natural_response = content.strip()
            coord = {}

        return {
            "response": natural_response,
            "coordinates": coord
        }

    except Exception as e:
        print(f"请求失败：{e}")
        return {"response": "处理失败", "coordinates": {}}


# ----------------------- SAM 分割相关 -----------------------

def choose_model():
    """初始化 SAM 分割预测器，设置相关参数"""
    model_weight = 'sam_b.pt'
    overrides = dict(
        task='segment',
        mode='predict',
        model=model_weight,
        conf=0.01,
        save=False
    )
    return SAMPredictor(overrides=overrides)


def process_sam_results(results):
    """处理 SAM 分割结果，获取掩码和中心点"""
    if not results or not results[0].masks:
        return None, None
    mask = results[0].masks.data[0].cpu().numpy()
    mask = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask
    M = cv2.moments(contours[0])
    if M["m00"] == 0:
        return None, mask
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), mask


# ----------------------- 语音识别与 TTS -----------------------

samplerate = 16000
channels = 1
dtype = 'int16'
frame_duration = 0.2
frame_samples = int(frame_duration * samplerate)
silence_threshold = 500
silence_max_duration = 1.0
q = queue.Queue()


def rms(audio_frame):
    samples = np.frombuffer(audio_frame, dtype=np.int16)
    if samples.size == 0:
        return 0
    mean_square = np.mean(samples.astype(np.float32) ** 2)
    if np.isnan(mean_square) or mean_square < 1e-5:
        return 0
    return np.sqrt(mean_square)


def callback(indata, frames, time_info, status):
    if status:
        print("⚠️ 状态警告：", status)
    q.put(bytes(indata))


def recognize_speech():
    """录音并返回音频数据（numpy 数组）"""
    print("🎙️ 启动录音，请说话...")
    audio_buffer = []
    is_speaking = False
    last_voice_time = time.time()

    with sd.RawInputStream(samplerate=samplerate, blocksize=frame_samples,
                           dtype=dtype, channels=channels, callback=callback):
        while True:
            frame = q.get()
            volume = rms(frame)
            current_time = time.time()
            if volume > silence_threshold:
                if not is_speaking:
                    print("🎤 检测到语音，开始录音...")
                    is_speaking = True
                    audio_buffer = []
                audio_np = np.frombuffer(frame, dtype=np.int16)
                audio_buffer.append(audio_np)
                last_voice_time = current_time
            elif is_speaking and (current_time - last_voice_time > silence_max_duration):
                print("🛑 停止录音，准备识别...")
                full_audio = np.concatenate(audio_buffer, axis=0)
                return full_audio


def speech_to_text(audio_data):
    """
    将录音数据转换为文本：
    先将 numpy 数组保存为 wav，再转换为 mp3 发送至 ASR 接口。
    """
    ASR_API_URL = "http://192.168.1.6:3003/v1/audio/transcriptions"
    ASR_API_TOKEN = "tts-ncut1034"

    wav_io = io.BytesIO()
    write(wav_io, samplerate, audio_data.astype(np.int16))
    wav_io.seek(0)
    audio = AudioSegment.from_wav(wav_io)
    mp3_io = io.BytesIO()
    audio.export(mp3_io, format="mp3")
    mp3_io.seek(0)
    headers = {"Authorization": f"Bearer {ASR_API_TOKEN}"}
    files = {"file": ("speech.mp3", mp3_io, "audio/mpeg")}
    print("📡 正在识别语音...")
    response = requests.post(ASR_API_URL, headers=headers, files=files)
    if response.ok:
        return response.text.strip()
    else:
        print("❌ 语音识别接口失败：", response.status_code)
        return ""


def play_tts(text):
    """
    调用 TTS 接口，将文本转换为语音并播放
    """
    TTS_API_URL = "http://192.168.1.6:3002/v1/audio/speech"
    TTS_API_TOKEN = "tts-ncut1034"
    print("📢 播放 TTS 语音：", text)
    payload = {
        "model": "cosyvoice",
        "voice": "tarzan",
        "input": text,
        "response_format": "mp3",
        "speed": 1.2
    }
    headers = {
        "Authorization": f"Bearer {TTS_API_TOKEN}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(TTS_API_URL, headers=headers, data=json.dumps(payload))
        if response.ok:
            audio_data = response.content
            audio_io = io.BytesIO(audio_data)
            audio_seg = AudioSegment.from_file(audio_io, format="mp3")
            raw = np.array(audio_seg.get_array_of_samples())
            raw = raw.reshape((-1, audio_seg.channels))
            sd.play(raw, audio_seg.frame_rate)
            sd.wait()
        else:
            print("❌ TTS 接口失败：", response.status_code)
    except Exception as e:
        print("❌ 播报失败：", e)


def voice_command_to_keyword():
    """
    获取语音命令并转换为文本。
    直接返回识别的文本指令。
    """
    audio_data = recognize_speech()
    text = speech_to_text(audio_data)
    if not text:
        print("⚠️ 没有识别到文本")
        return ""
    print("📝 识别文本：", text)
    return text


# ----------------------- 主流程：图像分割 -----------------------

def segment_image(image_input, output_mask='mask1.png'):
    """
    自动语音获取检测目标 → 多模态模型检测 → SAM 分割 → 保存掩码
    参数 image_input 为 numpy 数组（BGR 格式）。
    检测不到时支持手动点击选择目标区域。
    """
    # 1. 使用语音获取目标指令
    print("🎙️ 请通过语音描述目标物体及抓取指令...")
    command_text = voice_command_to_keyword()
    if not command_text:
        print("⚠️ 未识别到语音指令，请重试。")
        return None
    print(f"✅ 识别的语音指令：{command_text}")

    # 2. 通过多模态模型获取检测框
    result = generate_robot_actions(command_text, image_input)
    natural_response = result["response"]
    detection_info = result["coordinates"]
    print("自然语言回应：", natural_response)
    print("检测到的物体信息：", detection_info)

    # 仅对模型返回的自然语言回应播报
    play_tts(natural_response)

    bbox = detection_info.get("bbox") if detection_info and "bbox" in detection_info else None

    # 3. 准备图像供 SAM 使用（转换为 RGB）
    image_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)

    # 4. 初始化 SAM，并设置图像
    predictor = choose_model()
    predictor.set_image(image_rgb)

    if bbox:
        results = predictor(bboxes=[bbox])
        center, mask = process_sam_results(results)
        print(f"✅ 自动检测到目标，bbox：{bbox}")
    else:
        print("⚠️ 未检测到目标，请点击图像选择对象")
        cv2.namedWindow('Select Object', cv2.WINDOW_NORMAL)
        cv2.imshow('Select Object', image_input)
        point = []

        def click_handler(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                point.extend([x, y])
                print(f"🖱️ 点击坐标：{x}, {y}")
                cv2.setMouseCallback('Select Object', lambda *args: None)

        cv2.setMouseCallback('Select Object', click_handler)
        while True:
            key = cv2.waitKey(100)
            if point:
                break
            if cv2.getWindowProperty('Select Object', cv2.WND_PROP_VISIBLE) < 1:
                print("❌ 窗口被关闭，未进行点击")
                return None
        cv2.destroyAllWindows()
        results = predictor(points=[point], labels=[1])
        center, mask = process_sam_results(results)

    # 5. 保存分割掩码
    if mask is not None:
        cv2.imwrite(output_mask, mask, [cv2.IMWRITE_PNG_BILEVEL, 1])
        print(f"✅ 分割掩码已保存：{output_mask}")
    else:
        print("⚠️ 分割失败，未生成掩码")
    return mask


# ----------------------- 主程序入口 -----------------------

if __name__ == "__main__":
    # 示例：使用 cv2 读取图像，并以 numpy 数组传入
    input_image = cv2.imread('color.png')
    if input_image is None:
        raise ValueError("无法读取图像文件: color.png")
    seg_mask = segment_image(input_image)
    if seg_mask is not None:
        print("Segmentation result mask shape:", seg_mask.shape)
    else:
        print("Segmentation result: None")
