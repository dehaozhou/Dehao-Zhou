import requests
import json
import re


def generate_robot_actions(user_command):
    url = "http://localhost:1034/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    # 保留必要的“系统信息”提示，告诉模型基本需求。但不强制它只能回复纯 JSON
    system_prompt = """你是一个工业机械臂控制专家。请将用户指令解析为标准化动作序列。

预定义动作池：
[抓取] 参数：物体名称
[移动] 参数：目标位置 (坐标/语义位置)
[平移] 参数：方向(前/后/左/右)、距离(米)
[放置] 参数：放置位置
[倾斜] 参数：角度(1-90度)
[等待] 参数：秒数
[循环] 参数：次数

输出要求：
1. 尽量使用JSON数组格式描述动作序列
2. 自动补充合理参数值
3. 位置参数优先使用语义描述
4. 复杂操作拆解为多步骤
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_command}
    ]

    data = {
        "model": "Qwen3B",
        "messages": messages
        # 如果您有其他参数（如 temperature、stream 等），可自行添加
    }

    try:
        # 如果是自签名证书，需忽略验证，加上 verify=False
        response = requests.post(url, headers=headers, json=data, timeout=10)
        print("Status Code:", response.status_code)
        print("Response Text:", response.text)
        response.raise_for_status()

        # 首先尝试将整个响应解析为 JSON（OpenAI ChatCompletion 风格）
        result = response.json()

        # 获取模型返回的文本
        content = result["choices"][0]["message"]["content"]

        # 尝试直接把 content 作为 JSON 解析
        # 如果大模型确实给出了纯 JSON，就能直接解析成功
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # 如果直接解析失败，再去提取三反引号内的 JSON 片段
            pass

        # 用正则匹配三反引号包裹的 JSON
        pattern = r'```json\s*(.*?)\s*```'
        matches = re.findall(pattern, content, flags=re.DOTALL)
        if matches:
            # 如果找到多段，只解析第一段
            json_str = matches[0].strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # 如果也没有匹配到或解析失败，就返回空列表
        return []

    except Exception as e:
        print(f"请求失败：{e}")
        return []


if __name__ == "__main__":

    #test_commands = ["我渴了"]

    #test_commands = ["我想看你跳个舞"]

    #test_commands = ["我想吃面包"]

    test_commands = ["请帮我打开抽屉，然后关上抽屉"]

    for cmd in test_commands:
        print(f"输入指令：{cmd}")
        actions = generate_robot_actions(cmd)
        print("生成动作序列：")
        print(json.dumps(actions, indent=2, ensure_ascii=False))
        print("\n" + "=" * 50 + "\n")
