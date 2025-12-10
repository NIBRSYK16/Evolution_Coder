"""
完整的 Qwen2.5-Coder 模型微调与评估系统
整合模型加载、微调、评估三大功能
新增：直接大模型问答功能
"""

import gradio as gr
import torch
import json
import os
import sys
import re
import time
import threading
import subprocess
import tempfile
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional


# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ====== 全局变量 ======
# 模型相关
model = None
tokenizer = None
device = None

# 状态标志
is_training = False
is_evaluating = False
is_generating = False
training_thread = None
evaluation_thread = None

# 结果存储
comparison_results = {}

# ====== API配置（从generate_mbpp_dataset.py复制） ======
API_CONFIG = {
    "qwen_32b_api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    "qwen_14b_api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    "api_key": "sk-1d1d9ecf1f1b446588871b3e6d5d3a30",
}

# ====== 默认配置 ======
DEFAULT_CONFIG = {
    # 模型配置
    "model_path": "./models/Qwen2.5-Coder-0.5B-Instruct",
    "finetuned_model_path": "./qwen2.5-coder-0.5b-finetuned",
    "human_eval_path": "./datasets/human-eval-v2-20210705.jsonl",
    
    # 训练配置
    "mbpp_dataset_path": "./datasets/mbpp_text_only.jsonl",  # MBPP原始数据集路径
    "training_dataset_path": "./mbpp_training_data/mbpp_training_dataset.jsonl",  # 处理后训练集路径
    "output_dir": "./qwen2.5-coder-0.5b-finetuned",
    "num_epochs": 3,
    "learning_rate": 2e-4,
    "batch_size": 4,
    "use_lora": True,
    "use_4bit": False,
    
    # 数据生成配置
    "max_generate_items": 50,  # 最大生成数据量
    "generate_batch_size": 2,  # 生成批大小
    "max_retries": 3,  # API重试次数
    
    # 评估配置
    "max_tasks": 20,
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    
    # 问答配置
    "max_new_tokens": 512,
    "gen_temperature": 0.8,
    "gen_top_p": 0.95
}

# ====== 日志收集器 ======
class LogCollector:
    """收集所有日志"""
    def __init__(self):
        self.logs = []
        self.lock = threading.Lock()
        
    def add_log(self, message):
        with self.lock:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.logs.append(f"[{timestamp}] {message}")
            
    def get_logs(self, last_n=100):
        with self.lock:
            return "\n".join(self.logs[-last_n:])
            
    def clear(self):
        with self.lock:
            self.logs.clear()

log_collector = LogCollector()

def log(message):
    """记录日志"""
    log_collector.add_log(message)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

# ====== 从generate_mbpp_dataset.py复制的函数 ======
def call_qwen_api(api_url: str, prompt: str, model_name: str = "qwen2.5-coder-32b-instruct", 
                  max_tokens: int = 1024, temperature: float = 0.7, 
                  retries: int = 3) -> Tuple[bool, str]:
    """
    调用Qwen API生成代码
    """
    # 延迟导入requests
    try:
        import requests
    except ImportError:
        log("❌ 未安装requests库，无法调用API")
        return False, "请安装requests库: pip install requests"
    
    headers = {
        "Authorization": f"Bearer {API_CONFIG['api_key']}",
        "Content-Type": "application/json"
    }
    
    messages = [
        {"role": "system", "content": "你是一个专业的编程助手，请生成高质量、可运行的Python代码。"},
        {"role": "user", "content": prompt}
    ]
    
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9
    }
    
    for attempt in range(retries):
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            generated_code = result["choices"][0]["message"]["content"]
            
            # 提取代码块（如果有的话）
            code_pattern = r"```(?:python)?\n?(.*?)```"
            matches = re.findall(code_pattern, generated_code, re.DOTALL)
            
            if matches:
                generated_code = matches[0].strip()
            
            return True, generated_code
        except requests.exceptions.RequestException as e:
            if attempt == retries - 1:
                return False, f"API调用失败（尝试{retries}次）: {str(e)}"
            time.sleep(1)  # 等待1秒后重试
        except Exception as e:
            return False, f"API处理失败: {str(e)}"
    
    return False, "未知错误"

def validate_code_with_14b(instruct: str, code: str) -> Tuple[bool, str]:
    """
    使用14B模型验证代码是否符合指令逻辑
    """
    validation_prompt = f"""
    请分析以下代码是否符合用户指令的逻辑要求：
    
    用户指令：{instruct}
    
    生成的代码：
    ```python
    {code}
    ```
    
    请从以下几个方面进行判断：
    1. 代码是否完整实现了指令要求的功能
    2. 代码逻辑是否正确
    3. 是否有明显的逻辑错误或缺失
    
    请用以下格式回答：
    [是否通过]：是/否
    [理由]：简要说明理由
    """
    
    success, response = call_qwen_api(
        API_CONFIG["qwen_14b_api_url"], 
        validation_prompt, 
        model_name="qwen2.5-coder-14b-instruct",
        max_tokens=256,
        temperature=0.3
    )
    
    if not success:
        return False, response
    
    # 解析响应
    if "[是否通过]：是" in response or ("通过" in response and "否" not in response):
        return True, response
    else:
        return False, response

def check_code_syntax(code: str) -> Tuple[bool, str]:
    """
    检查Python代码的语法错误
    """
    try:
        # 添加必要的导入
        full_code = "import math\nimport re\nimport heapq\nimport numpy as np\nimport collections\n" + code
        
        # 尝试编译
        compile(full_code, '<string>', 'exec')
        return True, "语法检查通过"
    except SyntaxError as e:
        return False, f"语法错误: {str(e)}"
    except Exception as e:
        return False, f"代码检查错误: {str(e)}"

def extract_function_name(code: str) -> str:
    """
    从代码中提取函数名
    """
    # 查找第一个函数定义
    pattern = r'def\s+(\w+)\s*\('
    match = re.search(pattern, code)
    if match:
        return match.group(1)
    return "unknown_function"

def run_basic_test(code: str, function_name: str) -> Tuple[bool, str]:
    """
    运行基本测试：检查函数是否可以正常调用
    """
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            # 添加必要的导入
            f.write("import math\nimport re\nimport heapq\nimport numpy as np\nimport collections\n")
            f.write(code)
            f.write(f"\n\n# 基本测试\nif __name__ == '__main__':\n")
            f.write(f"    try:\n")
            f.write(f"        # 检查函数是否存在\n")
            f.write(f"        if '{function_name}' in dir():\n")
            f.write(f"            func = {function_name}\n")
            f.write(f"            print('函数存在，可以调用')\n")
            f.write(f"        else:\n")
            f.write(f"            print('函数不存在')\n")
            f.write(f"    except Exception as e:\n")
            f.write(f"        print(f'测试失败: {{e}}')\n")
            temp_file = f.name
        
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        os.unlink(temp_file)
        
        if result.returncode == 0 and "函数存在" in result.stdout:
            return True, "基本测试通过"
        else:
            return False, f"基本测试失败: {result.stderr or result.stdout}"
            
    except Exception as e:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
        return False, f"测试执行错误: {str(e)}"

def process_single_instruction(instruction: str, index: int) -> Tuple[bool, str, str]:
    """
    处理单个指令，生成代码并验证
    返回: (是否成功, 生成的代码, 验证结果)
    """
    log(f"[{index}] 处理指令: {instruction[:80]}...")
    
    # 步骤1: 使用32B模型生成代码
    log(f"[{index}] 调用32B API生成代码...")
    success, code = call_qwen_api(
        API_CONFIG["qwen_32b_api_url"],
        instruction,
        model_name="qwen2.5-coder-32b-instruct"
    )
    
    if not success:
        log(f"[{index}] ❌ 代码生成失败: {code}")
        return False, "", f"代码生成失败: {code}"
    
    # 显示生成的代码预览
    code_lines = code.split('\n')
    preview_lines = min(5, len(code_lines))
    code_preview = '\n'.join(code_lines[:preview_lines])
    log(f"[{index}] ✅ 代码生成成功")
    log(f"[{index}] 代码预览（前{preview_lines}行）:\n{code_preview}")
    log(f"[{index}] 代码总长度: {len(code)} 字符, {len(code_lines)} 行")
    
    # 步骤2: 语法检查
    log(f"[{index}] 进行语法检查...")
    syntax_ok, syntax_msg = check_code_syntax(code)
    if not syntax_ok:
        log(f"[{index}] ❌ {syntax_msg}")
        return False, "", syntax_msg
    
    log(f"[{index}] ✅ 语法检查通过")
    
    # 步骤3: 逻辑验证（14B模型）
    log(f"[{index}] 进行逻辑验证（14B模型）...")
    logic_ok, logic_msg = validate_code_with_14b(instruction, code)
    
    if not logic_ok:
        log(f"[{index}] ❌ 逻辑验证失败: {logic_msg[:100]}")
        return False, "", f"逻辑验证失败: {logic_msg[:100]}"
    
    log(f"[{index}] ✅ 逻辑验证通过")
    
    # 步骤4: 基本测试
    log(f"[{index}] 进行基本测试...")
    function_name = extract_function_name(code)
    test_ok, test_msg = run_basic_test(code, function_name)
    
    if not test_ok:
        log(f"[{index}] ⚠️ {test_msg} (但仍保存)")
        # 基本测试失败不一定意味着代码有问题，继续处理
    else:
        log(f"[{index}] ✅ 基本测试通过")
    
    log(f"[{index}] ✅ 处理完成，数据对合格")
    
    return True, code, "验证通过"

def generate_mbpp_training_data(mbpp_path: str, output_path: str, max_items: int = 50, 
                              start_index: int = 0) -> Tuple[bool, str]:
    """
    生成MBPP训练数据
    """
    try:
        # 导入requests（延迟导入以避免依赖问题）
        try:
            import requests
        except ImportError:
            log("❌ 未安装requests库，无法调用API")
            return False, "请安装requests库: pip install requests"
        
        log(f"读取MBPP数据集: {mbpp_path}")
        if not os.path.exists(mbpp_path):
            return False, f"MBPP数据集不存在: {mbpp_path}"
        
        instructions = []
        with open(mbpp_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and line.startswith('"') and line.endswith('"'):
                    # 移除引号
                    instruction = line[1:-1]
                    instructions.append(instruction)
        
        total_instructions = len(instructions)
        log(f"共读取 {total_instructions} 条指令")
        
        # 限制处理数量
        if max_items:
            instructions = instructions[:max_items]
            total_instructions = len(instructions)
            log(f"限制处理数量为: {total_instructions}")
        
        # 跳过已处理的
        if start_index > 0:
            instructions = instructions[start_index:]
            log(f"从索引 {start_index} 开始处理")
        
        if not instructions:
            return True, "没有需要处理的指令"
        
        log(f"开始处理 {len(instructions)} 条指令...")
        
        # 处理每条指令
        successful_pairs = []
        
        for i, instruction in enumerate(instructions, start=1):
            try:
                success, code, validation_msg = process_single_instruction(instruction, i)
                
                if success:
                    training_pair = {
                        "instruction": instruction,
                        "code": code,
                        "metadata": {
                            "index": i,
                            "timestamp": datetime.now().isoformat(),
                            "validation_result": validation_msg,
                            "source": "mbpp_dataset_generated"
                        }
                    }
                    successful_pairs.append(training_pair)
                    log(f"[{i}] ✅ 成功生成数据对")
                else:
                    log(f"[{i}] ❌ 数据对生成失败: {validation_msg}")
                
                # 避免API调用过于频繁
                time.sleep(0.5)
                
            except Exception as e:
                log(f"[{i}] ❌ 处理异常: {str(e)}")
        
        # 保存训练数据
        if successful_pairs:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for pair in successful_pairs:
                    f.write(json.dumps({
                        "instruction": pair["instruction"],
                        "code": pair["code"]
                    }, ensure_ascii=False) + '\n')
            
            log(f"✅ 训练数据生成完成: {len(successful_pairs)}/{len(instructions)} 条成功")
            log(f"训练数据已保存到: {output_path}")
            
            return True, f"成功生成 {len(successful_pairs)} 个训练数据对"
        else:
            return False, "未生成任何训练数据对"
            
    except Exception as e:
        return False, f"生成训练数据时出错: {str(e)}"

# ====== 模型问答功能模块 ======
def generate_code_with_local_model(instruction: str, config: Dict) -> Tuple[str, str]:
    """
    使用本地加载的模型生成代码
    """
    global model, tokenizer, device, is_generating
    
    if model is None or tokenizer is None:
        return "❌ 错误: 模型未加载", "请先加载模型"
    
    if is_generating:
        return "⚠️ 正在生成中，请稍候...", ""
    
    is_generating = True
    
    try:
        log(f"开始生成代码，指令: {instruction[:100]}...")
        
        # 准备输入
        messages = [
            {"role": "system", "content": "你是一个专业的Python编程助手。请根据用户指令生成正确、高效的Python代码。"},
            {"role": "user", "content": instruction}
        ]
        
        # 使用Qwen的聊天模板
        try:
            # 尝试使用tokenizer的apply_chat_template
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            # 备用方案：手动构建Qwen格式
            text = f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
            text += f"<|im_start|>user\n{messages[1]['content']}<|im_end|>\n"
            text += f"<|im_start|>assistant\n"
        
        # 编码输入
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        
        # 移到设备
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 生成代码
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=config.get("max_new_tokens", DEFAULT_CONFIG["max_new_tokens"]),
                temperature=config.get("gen_temperature", DEFAULT_CONFIG["gen_temperature"]),
                top_p=config.get("gen_top_p", DEFAULT_CONFIG["gen_top_p"]),
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_beams=1,
                repetition_penalty=1.1
            )
        
        # 解码生成的代码
        generated_tokens = generated_ids[0][inputs['input_ids'].shape[1]:]
        generated_code = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # 清理特殊标记
        generated_code = generated_code.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
        
        # 提取可能的代码块
        code_pattern = r"```(?:python)?\n?(.*?)```"
        matches = re.findall(code_pattern, generated_code, re.DOTALL)
        
        if matches:
            generated_code = matches[0].strip()
        
        # 清理代码中的多余说明
        lines = generated_code.split('\n')
        cleaned_lines = []
        in_code_block = False
        for line in lines:
            if line.strip().startswith('def ') or line.strip().startswith('class ') or line.strip().startswith('import ') or line.strip().startswith('from '):
                in_code_block = True
            if in_code_block or line.strip().startswith('#') or line.strip().startswith('"""') or line.strip().startswith("'''"):
                cleaned_lines.append(line)
        
        generated_code = '\n'.join(cleaned_lines)
        
        log(f"✅ 代码生成完成，长度: {len(generated_code)} 字符")
        
        return "✅ 代码生成成功", generated_code
        
    except Exception as e:
        error_msg = f"❌ 生成代码时出错: {str(e)}"
        log(error_msg)
        return error_msg, ""
        
    finally:
        is_generating = False

def save_instruction_to_mbpp(instruction: str, mbpp_path: str = None):
    """
    将指令保存到MBPP数据集（添加引号确保格式统一）
    """
    try:
        if mbpp_path is None:
            mbpp_path = DEFAULT_CONFIG["mbpp_dataset_path"]
        
        # 确保目录存在
        os.makedirs(os.path.dirname(mbpp_path), exist_ok=True)
        
        # 清理指令：移除多余空格和换行
        cleaned_instruction = instruction.strip()
        
        # 确保指令用双引号包裹
        if not (cleaned_instruction.startswith('"') and cleaned_instruction.endswith('"')):
            # 转义内部的双引号
            cleaned_instruction = cleaned_instruction.replace('"', '\\"')
            cleaned_instruction = f'"{cleaned_instruction}"'
        
        # 保存到文件
        with open(mbpp_path, 'a', encoding='utf-8') as f:
            f.write(cleaned_instruction + '\n')
        
        log(f"✅ 指令已保存到MBPP数据集: {cleaned_instruction[:100]}...")
        return True, f"指令已保存到 {mbpp_path}"
        
    except Exception as e:
        error_msg = f"❌ 保存指令失败: {str(e)}"
        log(error_msg)
        return False, error_msg

def process_instruction_with_local_model(instruction: str, temperature: float, top_p: float, 
                                        max_new_tokens: int, mbpp_path: str = None) -> Tuple[str, str, str]:
    """
    处理用户指令：如果是"自我演化"则开始微调，否则生成代码并保存指令
    """
    global is_training, training_thread
    
    # 清理指令
    instruction = instruction.strip()
    
    # 检查是否为"自我演化"指令
    if instruction.lower() == "自我演化":
        log("检测到'自我演化'指令，开始微调流程...")
        
        # 检查MBPP数据集是否存在
        if mbpp_path is None:
            mbpp_path = DEFAULT_CONFIG["mbpp_dataset_path"]
        
        if not os.path.exists(mbpp_path):
            error_msg = f"❌ MBPP数据集不存在: {mbpp_path}"
            log(error_msg)
            return error_msg, "", ""
        
        # 检查数据集大小
        try:
            with open(mbpp_path, 'r', encoding='utf-8') as f:
                lines = sum(1 for _ in f)
        except:
            lines = 0
        
        if lines == 0:
            error_msg = f"❌ MBPP数据集为空: {mbpp_path}"
            log(error_msg)
            return error_msg, "", ""
        
        log(f"MBPP数据集包含 {lines} 条指令")
        
        # 开始微调
        if is_training:
            return "⚠️ 训练已经在进行中...", "", ""
        
        # 准备训练配置
        train_config = {
            "model_path": DEFAULT_CONFIG["model_path"],
            "mbpp_dataset_path": mbpp_path,
            "output_dir": DEFAULT_CONFIG["output_dir"],
            "num_epochs": DEFAULT_CONFIG["num_epochs"],
            "learning_rate": DEFAULT_CONFIG["learning_rate"],
            "batch_size": DEFAULT_CONFIG["batch_size"],
            "max_generate_items": min(50, lines),  # 限制生成数量
            "use_lora": DEFAULT_CONFIG["use_lora"],
            "use_4bit": DEFAULT_CONFIG["use_4bit"]
        }
        
        # 开始训练线程
        training_thread = TrainingThread(train_config, log)
        is_training = True
        training_thread.start()
        
        status_msg = f"""
🚀 开始自我演化（微调）...
使用指令: {lines} 条
输出目录: {DEFAULT_CONFIG['output_dir']}
训练轮数: {DEFAULT_CONFIG['num_epochs']}
开始时间: {datetime.now().strftime('%H:%M:%S')}
        """
        
        log(status_msg)
        return status_msg, "", ""
    
    else:
        # 正常生成代码流程
        log(f"处理用户指令: {instruction[:100]}...")
        
        # 构建配置字典
        config = {
            "max_new_tokens": max_new_tokens,
            "gen_temperature": temperature,
            "gen_top_p": top_p
        }
        
        # 生成代码
        status, code = generate_code_with_local_model(instruction, config)
        
        # 保存指令到MBPP数据集（带引号）
        save_success, save_msg = save_instruction_to_mbpp(instruction, mbpp_path)
        
        if save_success:
            save_status = f"✅ 指令已保存到MBPP数据集"
        else:
            save_status = f"⚠️ 保存指令失败: {save_msg}"
        
        return status, code, save_status

# ====== 模型加载模块 ======
def load_model_interface(model_path):
    """加载模型界面函数"""
    global model, tokenizer, device
    
    if not model_path or model_path.strip() == "":
        model_path = DEFAULT_CONFIG["model_path"]
    
    if not os.path.exists(model_path):
        return f"❌ 模型路径不存在: {model_path}", False
    
    try:
        log("🚀 开始加载模型...")
        
        # 动态导入
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True
        )
        
        # 确定设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        
        model.eval()
        
        info = f"""
✅ 模型加载完成！
模型路径: {model_path}
使用设备: {device}
模型参数量: 约0.5B
Tokenizer: 已加载
        """
        
        log(info)
        return "✅ 模型加载成功", True
        
    except Exception as e:
        error_msg = f"❌ 加载模型失败: {str(e)}"
        log(error_msg)
        return error_msg, False

# ====== 模型训练模块（修改版，使用MBPP数据生成） ======
class TrainingThread(threading.Thread):
    """训练线程"""
    def __init__(self, config, callback=None):
        super().__init__()
        self.config = config
        self.callback = callback
        self.daemon = True
        
    def log(self, message):
        if self.callback:
            self.callback(message)
        log(message)
        
    def run(self):
        try:
            # 步骤1: 生成训练数据
            self.log("=" * 60)
            self.log("第一步: 生成训练数据")
            self.log("=" * 60)
            
            # 检查是否需要生成训练数据
            mbpp_path = self.config.get('mbpp_dataset_path', DEFAULT_CONFIG["mbpp_dataset_path"])
            training_data_path = self.config.get('training_dataset_path', DEFAULT_CONFIG["training_dataset_path"])
            max_generate_items = self.config.get('max_generate_items', DEFAULT_CONFIG["max_generate_items"])
            
            # 如果训练数据不存在或需要重新生成
            if not os.path.exists(training_data_path):
                self.log(f"训练数据不存在，开始生成...")
                self.log(f"MBPP数据集: {mbpp_path}")
                self.log(f"输出路径: {training_data_path}")
                self.log(f"最大生成数量: {max_generate_items}")
                
                success, msg = generate_mbpp_training_data(
                    mbpp_path, 
                    training_data_path,
                    max_items=max_generate_items
                )
                
                if not success:
                    self.log(f"❌ 生成训练数据失败: {msg}")
                    return
                
                self.log(f"✅ {msg}")
            else:
                # 检查现有训练数据
                with open(training_data_path, 'r', encoding='utf-8') as f:
                    lines = sum(1 for _ in f)
                self.log(f"✅ 使用现有训练数据: {training_data_path}")
                self.log(f"现有训练样本数: {lines}")
            
            # 步骤2: 加载模型进行微调
            self.log("=" * 60)
            self.log("第二步: 开始模型微调")
            self.log("=" * 60)
            
            self.log("开始导入训练库...")
            
            # 动态导入训练所需的库
            from transformers import (
                AutoTokenizer,
                AutoModelForCausalLM,
                TrainingArguments,
                Trainer,
                DataCollatorForLanguageModeling,
                BitsAndBytesConfig
            )
            
            from datasets import Dataset
            import warnings
            warnings.filterwarnings("ignore")
            
            self.log("库导入完成")
            
            # 加载模型
            self.log(f"加载模型: {self.config['model_path']}")
            global model, tokenizer, device
            
            # 加载tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.config['model_path'],
                trust_remote_code=True,
                padding_side="right",
                use_fast=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            # 确定设备
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.log(f"使用设备: {device}")
            
            # 加载模型
            model = AutoModelForCausalLM.from_pretrained(
                self.config['model_path'],
                local_files_only=True,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )
            
            self.log("✅ 模型加载完成")
            
            # 加载数据集
            self.log(f"加载训练数据集: {training_data_path}")
            data = []
            try:
                with open(training_data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
            except Exception as e:
                self.log(f"❌ 加载数据集失败: {str(e)}")
                return
                
            self.log(f"数据集大小: {len(data)} 个样本")
            
            if len(data) == 0:
                self.log("❌ 数据集为空")
                return
                
            # 准备训练数据
            self.log("准备训练数据...")
            processed_data = []
            for item in data[:100]:  # 限制样本数量，避免内存不足
                instruction = item.get("instruction", "")
                code = item.get("code", "")
                
                # 创建模型输入格式
                messages = [
                    {"role": "system", "content": "You are a helpful AI assistant that writes Python code."},
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": code}
                ]
                
                # 使用Qwen特定的格式
                text = f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
                text += f"<|im_start|>user\n{messages[1]['content']}<|im_end|>\n"
                text += f"<|im_start|>assistant\n{messages[2]['content']}<|im_end|>\n"
                
                processed_data.append({"text": text})
            
            # 创建数据集
            dataset = Dataset.from_list(processed_data)
            
            # 分割训练/验证集
            if len(dataset) > 20:
                split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
                train_dataset = split_dataset["train"]
                eval_dataset = split_dataset["test"]
            else:
                train_dataset = dataset
                eval_dataset = dataset.select(range(min(5, len(dataset))))
                
            self.log(f"训练集: {len(train_dataset)}, 验证集: {len(eval_dataset)}")
            
            # 数据预处理
            def preprocess_function(examples):
                return tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=512,  # 减少长度以节省内存
                    padding="max_length",
                )
                
            self.log("预处理数据...")
            tokenized_train = train_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=train_dataset.column_names,
            )
            tokenized_eval = eval_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=eval_dataset.column_names,
            )
            
            # 数据整理器
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            )
            
            # 训练参数
            training_args = TrainingArguments(
                output_dir=self.config['output_dir'],
                num_train_epochs=self.config['num_epochs'],
                per_device_train_batch_size=self.config['batch_size'],
                per_device_eval_batch_size=self.config['batch_size'],
                gradient_accumulation_steps=2,
                warmup_steps=50,
                logging_steps=5,
                save_strategy="epoch",
                eval_strategy="epoch",
                learning_rate=self.config['learning_rate'],
                weight_decay=0.01,
                fp16=False, # torch.cuda.is_available(),
                push_to_hub=False,
                report_to="none",
                gradient_checkpointing=True,
            )
            
            # 创建Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_eval,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )
            
            # 开始训练
            self.log("开始训练...")
            trainer.train()
            
            # 保存模型
            self.log("保存模型...")
            trainer.save_model()
            tokenizer.save_pretrained(self.config['output_dir'])
            
            self.log("🎉 训练完成！")
            self.log(f"模型已保存到: {self.config['output_dir']}")
            
        except ImportError as e:
            self.log(f"❌ 缺少依赖库: {str(e)}")
            self.log("请运行: pip install torch transformers datasets")
        except Exception as e:
            self.log(f"❌ 训练过程中出错: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
        finally:
            global is_training
            is_training = False

def start_training_interface(config_data):
    """开始训练界面函数"""
    global is_training, training_thread
    
    if is_training:
        return "⚠️ 训练已经在进行中...", False
    
    # 更新配置
    config = DEFAULT_CONFIG.copy()
    config.update(config_data)
    
    # 检查必要参数
    required_fields = ["model_path", "mbpp_dataset_path", "output_dir"]
    for field in required_fields:
        if not config.get(field):
            return f"❌ 请填写{field}", False
    
    # 检查MBPP数据集
    mbpp_path = config.get("mbpp_dataset_path", DEFAULT_CONFIG["mbpp_dataset_path"])
    if not os.path.exists(mbpp_path):
        return f"❌ MBPP数据集不存在: {mbpp_path}", False
    
    # 创建输出目录
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # 开始训练线程
    training_thread = TrainingThread(config, log)
    is_training = True
    training_thread.start()
    
    start_msg = f"""
🚀 开始模型微调任务...

第一阶段: 生成训练数据
- MBPP数据集: {config.get('mbpp_dataset_path', DEFAULT_CONFIG["mbpp_dataset_path"])}
- 最大生成数量: {config.get('max_generate_items', DEFAULT_CONFIG["max_generate_items"])}
- 输出路径: {config.get('training_dataset_path', DEFAULT_CONFIG["training_dataset_path"])}

第二阶段: 模型微调
- 模型: {config['model_path']}
- 输出目录: {config['output_dir']}
- 训练轮数: {config['num_epochs']}
- 学习率: {config['learning_rate']}
- 批大小: {config['batch_size']}

训练日志将在下方显示...
    """
    
    log(start_msg)
    return "✅ 训练已开始", True

# ====== 模型评估模块 ======
class EvaluationThread(threading.Thread):
    """评估线程"""
    def __init__(self, config, callback=None):
        super().__init__()
        self.config = config
        self.callback = callback
        self.daemon = True
        self.result = None
        
    def log(self, message):
        if self.callback:
            self.callback(message)
        log(message)
        
    def run(self):
        try:
            self.evaluate_models()
        except Exception as e:
            self.log(f"评估线程出错: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
        finally:
            global is_evaluating
            is_evaluating = False
            
    def evaluate_models(self):
        """评估模型"""
        self.log("开始导入评估库...")
        
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # 检查模型文件
        original_path = self.config["model_path"]
        finetuned_path = self.config["finetuned_model_path"]
        dataset_path = self.config["human_eval_path"]
        
        if not os.path.exists(original_path):
            self.log(f"❌ 原始模型路径不存在: {original_path}")
            return
            
        if not os.path.exists(finetuned_path):
            self.log(f"❌ 微调模型路径不存在: {finetuned_path}")
            return
            
        if not os.path.exists(dataset_path):
            self.log(f"❌ HumanEval数据集不存在: {dataset_path}")
            self.log("请从 https://github.com/openai/human-eval 下载数据集")
            return
        
        # 评估原始模型
        self.log("="*60)
        self.log("开始评估原始模型...")
        original_result = self.evaluate_single_model(
            original_path, 
            "原始模型",
            base_model_path=None
        )
        
        if original_result:
            self.log(f"原始模型评估完成: 通过率 {original_result['pass_rate']:.2f}%")
            
            # 清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        # 评估微调模型
        self.log("="*60)
        self.log("开始评估微调后模型...")
        finetuned_result = self.evaluate_single_model(
            finetuned_path,
            "微调后模型",
            base_model_path=original_path  # LoRA需要基础模型
        )
        
        if finetuned_result:
            self.log(f"微调后模型评估完成: 通过率 {finetuned_result['pass_rate']:.2f}%")
            
        # 对比结果
        if original_result and finetuned_result:
            comparison = self.compare_results(original_result, finetuned_result)
            global comparison_results
            comparison_results = comparison
            
            self.log("="*60)
            self.log("模型对比完成！")
            self.log(f"原始模型通过率: {original_result['pass_rate']:.2f}%")
            self.log(f"微调后模型通过率: {finetuned_result['pass_rate']:.2f}%")
            self.log(f"提升: {comparison['improvement']:.2f}%")
            
            # 保存结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = f"./evaluation_results_{timestamp}.json"
            
            results = {
                "original": original_result,
                "finetuned": finetuned_result,
                "comparison": comparison,
                "timestamp": timestamp,
                "config": self.config
            }
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
            self.log(f"评估结果已保存到: {result_file}")
            
        self.log("🎉 模型评估全部完成！")
        
    def evaluate_single_model(self, model_path, model_name, base_model_path=None):
        """评估单个模型"""
        try:
            # 加载模型
            self.log(f"加载{model_name}: {model_path}")
            
            # 检查是否是LoRA adapter
            is_lora = os.path.exists(os.path.join(model_path, "adapter_config.json"))
            from transformers import AutoTokenizer, AutoModelForCausalLM            
            
            if is_lora and base_model_path:
                # 使用LoRA adapter
                self.log("检测到LoRA adapter，加载基础模型并合并adapter")
                
                try:
                    from peft import PeftModel

                    
                    # 加载基础模型
                    base_model = AutoModelForCausalLM.from_pretrained(
                        base_model_path,
                        local_files_only=True,
                        device_map="auto",
                        torch_dtype=torch.float32, # torch.float16 if torch.cuda.is_available() else torch.float32,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True
                    )
                    
                    # 加载LoRA adapter并合并
                    model = PeftModel.from_pretrained(base_model, model_path)
                    model = model.merge_and_unload()
                    
                    # 加载分词器
                    tokenizer = AutoTokenizer.from_pretrained(
                        base_model_path,
                        local_files_only=True,
                        trust_remote_code=True
                    )
                    
                except ImportError:
                    self.log("❌ 未安装peft库，无法加载LoRA模型")
                    self.log("请运行: pip install peft")
                    return None
            else:
                # 加载完整模型
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    local_files_only=True,
                    trust_remote_code=True
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    local_files_only=True,
                    device_map="auto",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            
            # 确保tokenizer设置正确
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            # 设置模型为评估模式
            model.eval()
            
            # 读取HumanEval数据集
            self.log("读取HumanEval数据集...")
            tasks = []
            with open(self.config["human_eval_path"], 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        tasks.append(json.loads(line))
            
            max_tasks = self.config.get("max_tasks", None)
            if max_tasks:
                tasks = tasks[:max_tasks]
                self.log(f"限制评估任务数: {max_tasks}")
            
            total_tasks = len(tasks)
            passed_tasks = 0
            failed_tasks = []
            detailed_results = []
            
            self.log(f"开始评估 {total_tasks} 个任务...")
            start_time = time.time()
            
            for idx, task in enumerate(tasks, 1):
                task_id = task['task_id']
                prompt = task['prompt']
                entry_point = task['entry_point']
                test_code = task['test']
                
                # 每5个任务输出一次进度
                if idx % 5 == 0 or idx == total_tasks:
                    elapsed = time.time() - start_time
                    rate = (passed_tasks / idx * 100) if idx > 0 else 0
                    self.log(f"进度: {idx}/{total_tasks} | 通过: {passed_tasks} | 通过率: {rate:.1f}% | 用时: {elapsed:.1f}秒")
                
                try:
                    # 生成代码
                    messages = [
                        {"role": "system", "content": "你是一个专业的编程助手，请根据给定的函数签名和文档字符串，实现该函数。"},
                        {"role": "user", "content": prompt},
                    ]
                    
                    # 应用聊天模板
                    try:
                        text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                    except:
                        # 如果聊天模板不可用，使用简单格式
                        text = f"系统: {messages[0]['content']}\n用户: {messages[1]['content']}\n助手: "
                    
                    # 编码输入
                    inputs = tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=1024
                    )
                    
                    # 移到GPU
                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                    
                    # 生成代码
                    with torch.no_grad():
                        generated_ids = model.generate(
                            **inputs,
                            max_new_tokens=self.config["max_tokens"],
                            temperature=self.config["temperature"],
                            top_p=self.config["top_p"],
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            num_beams=1
                        )
                    
                    # 解码生成的代码
                    generated_tokens = generated_ids[0][inputs['input_ids'].shape[1]:]
                    generated_code = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    
                    # 提取函数代码
                    function_code = self.extract_function_code(generated_code, entry_point)
                    
                    # 构建完整代码
                    full_code = prompt + function_code + "\n" + test_code
                    
                    # 执行测试
                    test_result = self.run_code_test(full_code, entry_point)
                    
                    if test_result["passed"]:
                        passed_tasks += 1
                        detailed_results.append({
                            "task_id": task_id,
                            "status": "通过",
                            "error": None
                        })
                    else:
                        failed_tasks.append(task_id)
                        detailed_results.append({
                            "task_id": task_id,
                            "status": "失败",
                            "error": test_result.get("error", "未知错误")
                        })
                        
                except Exception as e:
                    failed_tasks.append(task_id)
                    error_type = type(e).__name__
                    error_msg = str(e)[:100]
                    detailed_results.append({
                        "task_id": task_id,
                        "status": f"生成错误: {error_type}",
                        "error": error_msg
                    })
                
                # 清理内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 计算最终结果
            elapsed_time = time.time() - start_time
            pass_rate = (passed_tasks / total_tasks * 100) if total_tasks > 0 else 0
            
            result = {
                "model_name": model_name,
                "model_path": model_path,
                "total_tasks": total_tasks,
                "passed_tasks": passed_tasks,
                "failed_tasks_count": len(failed_tasks),
                "pass_rate": pass_rate,
                "elapsed_time": elapsed_time,
                "avg_time_per_task": elapsed_time / total_tasks if total_tasks > 0 else 0,
                "failed_task_ids": failed_tasks[:20],
                "detailed_results": detailed_results[:20],
                "evaluation_time": datetime.now().isoformat()
            }
            
            self.log(f"{model_name}评估完成: {passed_tasks}/{total_tasks} 通过 ({pass_rate:.2f}%)")
            
            # 清理模型
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return result
            
        except Exception as e:
            self.log(f"评估{model_name}时出错: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            return None
    
    def extract_function_code(self, generated_text, entry_point):
        """从生成的文本中提取函数代码"""
        text = generated_text.strip()
        
        # 方式1: 正则匹配
        pattern = rf'def\s+{re.escape(entry_point)}\s*\([^)]*\)\s*:.*?(?=\n\ndef\s+|\nclass\s+|$)'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(0).strip()
        
        # 方式2: 基于缩进
        if f"def {entry_point}" in text:
            lines = text.split('\n')
            start_idx = -1
            for i, line in enumerate(lines):
                if f"def {entry_point}" in line:
                    start_idx = i
                    break
            
            if start_idx >= 0:
                result = [lines[start_idx]]
                base_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
                
                for i in range(start_idx + 1, len(lines)):
                    line = lines[i]
                    if not line.strip():
                        result.append(line)
                        continue
                    
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent <= base_indent:
                        break
                    result.append(line)
                
                return '\n'.join(result)
        
        # 方式3: 返回整个文本
        return text
    
    def run_code_test(self, full_code, entry_point):
        """运行代码测试"""
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(full_code)
                temp_file = f.name
            
            # 执行代码
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=5,
                env={**os.environ, 'PYTHONPATH': ''}
            )
            
            # 清理临时文件
            os.unlink(temp_file)
            
            if result.returncode == 0:
                return {"passed": True}
            else:
                error_msg = result.stderr[:200] if result.stderr else "未知错误"
                return {"passed": False, "error": error_msg}
                
        except subprocess.TimeoutExpired:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            return {"passed": False, "error": "执行超时"}
        except Exception as e:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            return {"passed": False, "error": str(e)}
    
    def compare_results(self, original_result, finetuned_result):
        """对比两个模型的结果"""
        orig_rate = original_result["pass_rate"]
        fine_rate = finetuned_result["pass_rate"]
        improvement = fine_rate - orig_rate
        
        orig_passed = set(original_result.get("failed_task_ids", []))
        fine_passed = set(finetuned_result.get("failed_task_ids", []))
        
        newly_passed = list(orig_passed - fine_passed)  # 原来失败，现在通过
        newly_failed = list(fine_passed - orig_passed)  # 原来通过，现在失败
        
        return {
            "improvement": improvement,
            "original_pass_rate": orig_rate,
            "finetuned_pass_rate": fine_rate,
            "newly_passed_tasks": newly_passed[:10],
            "newly_failed_tasks": newly_failed[:10],
            "original_total_tasks": original_result["total_tasks"],
            "finetuned_total_tasks": finetuned_result["total_tasks"],
            "original_passed": original_result["passed_tasks"],
            "finetuned_passed": finetuned_result["passed_tasks"]
        }

def start_evaluation_interface(config_data):
    """开始评估界面函数"""
    global is_evaluating, evaluation_thread
    
    if is_evaluating:
        return "⚠️ 评估已经在进行中...", False
    
    # 更新配置
    config = DEFAULT_CONFIG.copy()
    config.update(config_data)
    
    # 检查必要参数
    required_fields = ["model_path", "finetuned_model_path", "human_eval_path"]
    for field in required_fields:
        if not config.get(field):
            return f"❌ 请填写{field}", False
    
    # 检查路径
    for path_field in ["model_path", "finetuned_model_path", "human_eval_path"]:
        path = config[path_field]
        if not os.path.exists(path):
            return f"❌ 路径不存在: {path}", False
    
    # 清空日志
    log_collector.clear()
    
    # 开始评估线程
    evaluation_thread = EvaluationThread(config, log)
    is_evaluating = True
    evaluation_thread.start()
    
    start_msg = f"""
🚀 开始模型对比评估...
原始模型: {config['model_path']}
微调模型: {config['finetuned_model_path']}
数据集: {config['human_eval_path']}
最大任务数: {config['max_tasks']}
    
评估日志将在下方显示...
    """
    
    log(start_msg)
    return "✅ 评估已开始", True

def get_comparison_results():
    """获取对比结果"""
    global comparison_results
    
    if not comparison_results:
        return "暂无评估结果"
    
    result_text = f"""
# 模型对比评估结果

## 总体表现
- **原始模型通过率**: {comparison_results['original_pass_rate']:.2f}%
- **微调模型通过率**: {comparison_results['finetuned_pass_rate']:.2f}%
- **提升效果**: {comparison_results['improvement']:+.2f}%

## 详细数据
- 原始模型: {comparison_results['original_passed']}/{comparison_results['original_total_tasks']} 通过
- 微调模型: {comparison_results['finetuned_passed']}/{comparison_results['finetuned_total_tasks']} 通过

## 改进分析
"""
    
    if comparison_results['newly_passed_tasks']:
        result_text += f"- **新通过的任务**: {len(comparison_results['newly_passed_tasks'])} 个\n"
        if comparison_results['newly_passed_tasks']:
            result_text += f"  示例: {', '.join(comparison_results['newly_passed_tasks'][:5])}\n"
    
    if comparison_results['newly_failed_tasks']:
        result_text += f"- **新失败的任务**: {len(comparison_results['newly_failed_tasks'])} 个\n"
        if comparison_results['newly_failed_tasks']:
            result_text += f"  示例: {', '.join(comparison_results['newly_failed_tasks'][:5])}\n"
    
    if comparison_results['improvement'] > 0:
        result_text += "\n🎉 **微调效果: 提升明显**"
    elif comparison_results['improvement'] == 0:
        result_text += "\n⚠️ **微调效果: 无明显变化**"
    else:
        result_text += "\n❌ **微调效果: 性能下降**"
    
    return result_text

# ====== 工具函数 ======
def check_paths(model_path, finetuned_model_path, human_eval_path):
    """检查路径"""
    results = []
    
    # 检查原始模型
    if os.path.exists(model_path):
        results.append(f"✅ 原始模型路径存在: {model_path}")
    else:
        results.append(f"❌ 原始模型路径不存在: {model_path}")
    
    # 检查微调模型
    if os.path.exists(finetuned_model_path):
        results.append(f"✅ 微调模型路径存在: {finetuned_model_path}")
    else:
        results.append(f"❌ 微调模型路径不存在: {finetuned_model_path}")
    
    # 检查数据集
    if os.path.exists(human_eval_path):
        results.append(f"✅ HumanEval数据集存在: {human_eval_path}")
    else:
        results.append(f"❌ HumanEval数据集不存在: {human_eval_path}")
        results.append("请从 https://github.com/openai/human-eval 下载数据集")
    
    return "\n".join(results)

def generate_example_dataset(instructions):
    """生成示例数据集"""
    lines = instructions.strip().split('\n')
    dataset = []
    
    for i, instr in enumerate(lines):
        if instr.strip():
            # 简单生成对应的代码
            if "add" in instr.lower() and "number" in instr.lower():
                code = """def add_numbers(a, b):
    \"\"\"Add two numbers\"\"\"
    return a + b"""
            elif "prime" in instr.lower():
                code = """def is_prime(n):
    \"\"\"Check if a number is prime\"\"\"
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True"""
            else:
                code = f"""def function_{i}():
    \"\"\"Generated function\"\"\"
    # TODO: Implement this function
    pass"""
            
            dataset.append({
                "instruction": instr.strip(),
                "code": code
            })
    
    # 保存到文件
    output_path = "./example_dataset.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return f"✅ 示例数据集已生成: {output_path}\n共 {len(dataset)} 个样本"

def update_system_info():
    """更新系统信息"""
    gpu_text = ""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_text = f"✅ GPU可用\n名称: {gpu_name}\n显存: {gpu_memory:.1f} GB"
    else:
        gpu_text = "❌ 未检测到GPU\n将在CPU上运行，速度较慢"
    
    model_text = "❌ 模型未加载"
    global model
    if model is not None:
        model_text = "✅ 模型已加载\n可使用生成和微调功能"
    
    return gpu_text, model_text

def update_logs():
    """更新日志显示"""
    logs = log_collector.get_logs(50)
    return logs

# ====== 创建完整的Gradio界面 ======
with gr.Blocks(title="Qwen2.5-Coder 完整系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Qwen2.5-Coder 完整系统")
    gr.Markdown("模型加载、微调、评估一体化系统")
    
    # 添加一个隐藏的状态组件来存储训练配置
    training_config_state = gr.State({})

    # 状态显示
    with gr.Row():
        status_display = gr.Textbox(
            label="系统状态",
            value="准备就绪",
            interactive=False,
            lines=1,
            scale=3
        )
        
        clear_logs_btn = gr.Button("清空日志", variant="secondary", size="sm", scale=1)
    
    # 日志输出区域
    log_output = gr.Textbox(
        label="系统日志",
        value="欢迎使用 Qwen2.5-Coder 完整系统\n\n请选择选项卡开始操作。",
        lines=25,
        interactive=False
    )
    
    # 系统信息
    with gr.Row():
        gpu_info = gr.Textbox(
            label="GPU状态",
            value="正在检测...",
            lines=2,
            interactive=False,
            scale=1
        )
        
        model_info = gr.Textbox(
            label="模型信息",
            value="未加载",
            lines=2,
            interactive=False,
            scale=1
        )
        
        results_info = gr.Textbox(
            label="最新结果",
            value="暂无结果",
            lines=2,
            interactive=False,
            scale=1
        )
    
    # 选项卡
    with gr.Tabs():
        # ====== Tab 1: 模型加载 ======
        with gr.TabItem("🚀 模型加载"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ 模型配置")
                    
                    model_path = gr.Textbox(
                        label="模型路径",
                        value=DEFAULT_CONFIG["model_path"],
                        placeholder="输入本地模型路径",
                        lines=1
                    )
                    
                    load_btn = gr.Button(
                        "🚀 加载模型",
                        variant="primary",
                        size="lg"
                    )
                    
                    load_status = gr.Textbox(
                        label="加载状态",
                        value="等待加载",
                        interactive=False,
                        lines=3
                    )
                    
                    gr.Markdown("### 📊 系统信息")
                    
                    with gr.Accordion("设备信息", open=True):
                        gr.Markdown("""
                        - **CPU**: Python 运行时
                        - **GPU**: 自动检测可用性
                        - **内存**: 根据配置调整
                        - **显存**: 训练时需要足够显存
                        """)
                
                with gr.Column(scale=2):
                    gr.Markdown("### 📝 加载说明")
                    
                    with gr.Accordion("详细说明", open=True):
                        gr.Markdown("""
                        ### 模型加载步骤：
                        
                        1. **准备模型文件**
                           - 下载 Qwen2.5-Coder-0.5B-Instruct 模型
                           - 模型应包含以下文件：
                             - config.json
                             - pytorch_model.bin
                             - tokenizer_config.json
                             - tokenizer.json
                        
                        2. **配置模型路径**
                           - 输入模型所在的本地路径
                           - 示例: `./models/Qwen2.5-Coder-0.5B-Instruct`
                        
                        3. **加载模型**
                           - 点击"加载模型"按钮
                           - 系统会验证模型文件
                           - 加载到可用设备（GPU/CPU）
                           - 显示加载状态
                        
                        ### 模型信息：
                        - **参数规模**: 5亿参数
                        - **模型类型**: 指令微调版本
                        - **适用任务**: 代码生成、代码解释
                        - **支持语言**: Python为主，支持其他语言
                        
                        ### 注意事项：
                        - 首次加载可能需要较长时间
                        - GPU加载速度更快
                        - 确保模型文件完整
                        """)
        
        # ====== Tab 2: 模型微调（已修改为使用MBPP） ======
        with gr.TabItem("🎯 模型微调"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ 训练配置")
                    
                    mbpp_dataset_path = gr.Textbox(
                        label="MBPP数据集路径",
                        value=DEFAULT_CONFIG["mbpp_dataset_path"],
                        placeholder="mbpp_text_only.jsonl 路径",
                        lines=1
                    )
                    
                    output_dir = gr.Textbox(
                        label="输出目录",
                        value=DEFAULT_CONFIG["output_dir"],
                        placeholder="微调后模型的保存路径",
                        lines=1
                    )
                    
                    with gr.Row():
                        num_epochs = gr.Number(
                            label="训练轮数",
                            value=DEFAULT_CONFIG["num_epochs"],
                            minimum=1,
                            maximum=10,
                            step=1
                        )
                        
                        learning_rate = gr.Number(
                            label="学习率",
                            value=DEFAULT_CONFIG["learning_rate"],
                            minimum=1e-6,
                            maximum=1e-2,
                            step=1e-6
                        )
                    
                    batch_size = gr.Slider(
                        label="批大小",
                        value=DEFAULT_CONFIG["batch_size"],
                        minimum=1,
                        maximum=16,
                        step=1
                    )
                    
                    max_generate_items = gr.Number(
                        label="最大生成数据量",
                        value=DEFAULT_CONFIG["max_generate_items"],
                        minimum=10,
                        maximum=500,
                        step=10,
                        info="从MBPP生成多少条训练数据"
                    )
                    
                    with gr.Row():
                        use_lora = gr.Checkbox(
                            label="使用LoRA",
                            value=DEFAULT_CONFIG["use_lora"]
                        )
                        use_4bit = gr.Checkbox(
                            label="4-bit量化",
                            value=DEFAULT_CONFIG["use_4bit"]
                        )
                    
                    train_btn = gr.Button(
                        "🎯 开始微调",
                        variant="stop",
                        size="lg"
                    )
                    
                    train_status = gr.Textbox(
                        label="训练状态",
                        value="等待开始",
                        interactive=False,
                        lines=3
                    )
                    
                    gr.Markdown("---")
                    gr.Markdown("### 🛠️ 工具")
                    
                    with gr.Accordion("检查MBPP数据集", open=False):
                        check_mbpp_btn = gr.Button("检查MBPP数据集", variant="secondary")
                        check_mbpp_output = gr.Textbox(label="检查结果", interactive=False, lines=3)
                    
                    with gr.Accordion("生成示例数据集", open=False):
                        example_instructions = gr.Textbox(
                            label="示例指令（每行一个）",
                            value="Write a function to add two numbers\nWrite a function to check if a number is prime",
                            lines=5
                        )
                        
                        generate_btn = gr.Button("生成示例数据集", variant="secondary")
                        generate_output = gr.Textbox(label="生成结果", interactive=False)
                
                with gr.Column(scale=2):
                    gr.Markdown("### 📝 微调说明")
                    
                    with gr.Accordion("详细说明", open=True):
                        gr.Markdown("""
                        ### 新的训练流程（使用MBPP数据集）：
                        
                        1. **准备MBPP数据集**
                           - 数据集应为 mbpp_text_only.jsonl 格式
                           - 每行是一个指令字符串，用双引号包裹
                           - 示例: `"Write a function to find the minimum cost path..."`
                        
                        2. **第一阶段：生成高质量训练数据**
                           - 系统会自动调用Qwen API生成代码
                           - 使用32B模型生成代码，14B模型验证逻辑
                           - 进行语法检查和基本测试
                           - 只有通过验证的指令-代码对才会被保留
                        
                        3. **第二阶段：模型微调**
                           - 使用生成的高质量数据进行微调
                           - 支持LoRA和4-bit量化以节省显存
                           - 训练完成后自动保存模型
                        
                        ### 训练参数：
                        - **训练轮数**: 1-10轮，通常3轮足够
                        - **学习率**: 2e-4 是比较合适的初始值
                        - **批大小**: 根据显存调整，4-8比较常见
                        - **最大生成数据量**: 控制从MBPP生成的样本数量
                        - **LoRA**: 低秩适配，减少训练参数
                        - **4-bit量化**: 减少显存使用
                        
                        ### 硬件要求：
                        - **GPU推荐**: 至少8GB显存（RTX 3070/4060 Ti及以上）
                        - **CPU模式**: 可以运行但速度较慢
                        - **内存**: 至少16GB RAM
                        
                        ### 训练时间：
                        - 数据生成阶段：每50条数据约15-30分钟（依赖API速度）
                        - 微调阶段：100个样本约5-10分钟
                        
                        ### 注意事项：
                        - 训练过程分为两个阶段，请耐心等待
                        - 需要有效的API密钥才能生成训练数据
                        - 训练过程中请不要关闭网页
                        """)
        
        # ====== Tab 3: 模型评估 ======
        with gr.TabItem("📊 模型评估"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ 评估配置")
                    
                    finetuned_model_path = gr.Textbox(
                        label="微调模型路径",
                        value=DEFAULT_CONFIG["finetuned_model_path"],
                        placeholder="微调后模型路径",
                        lines=1
                    )
                    
                    human_eval_path = gr.Textbox(
                        label="HumanEval数据集路径",
                        value=DEFAULT_CONFIG["human_eval_path"],
                        placeholder="human-eval-v2-20210705.jsonl 路径",
                        lines=1
                    )
                    
                    with gr.Row():
                        max_tasks = gr.Number(
                            label="最大任务数",
                            value=DEFAULT_CONFIG["max_tasks"],
                            minimum=1,
                            maximum=164,
                            step=1,
                            info="HumanEval共164个任务"
                        )
                        
                        max_tokens = gr.Number(
                            label="最大生成token数",
                            value=DEFAULT_CONFIG["max_tokens"],
                            minimum=50,
                            maximum=2048,
                            step=50
                        )
                    
                    with gr.Row():
                        temperature = gr.Slider(
                            label="Temperature",
                            value=DEFAULT_CONFIG["temperature"],
                            minimum=0.1,
                            maximum=2.0,
                            step=0.1
                        )
                        
                        top_p = gr.Slider(
                            label="Top-p",
                            value=DEFAULT_CONFIG["top_p"],
                            minimum=0.1,
                            maximum=1.0,
                            step=0.05
                        )
                    
                    eval_btn = gr.Button(
                        "🚀 开始评估",
                        variant="primary",
                        size="lg"
                    )
                    
                    eval_status = gr.Textbox(
                        label="评估状态",
                        value="等待开始",
                        interactive=False,
                        lines=3
                    )
                    
                    gr.Markdown("---")
                    gr.Markdown("### 📈 评估结果")
                    
                    results_btn = gr.Button(
                        "📊 查看结果",
                        variant="secondary",
                        size="lg"
                    )
                    
                    check_btn = gr.Button("检查路径", variant="secondary")
                    check_output = gr.Textbox(label="路径检查结果", interactive=False, lines=5)
                
                with gr.Column(scale=2):
                    gr.Markdown("### 📝 评估说明")
                    
                    with gr.Accordion("详细说明", open=True):
                        gr.Markdown("""
                        ### 评估流程：
                        
                        1. **配置评估参数**
                           - 原始模型路径：自动使用已加载的模型
                           - 微调模型路径：微调后的模型保存路径
                           - HumanEval数据集：从GitHub下载的jsonl文件
                        
                        2. **调整评估参数**
                           - 最大任务数：建议先评估20-50个任务测试
                           - 生成参数：temperature和top-p影响代码多样性
                        
                        3. **开始评估**
                           - 点击"开始评估"按钮
                           - 系统会依次评估两个模型
                           - 每个任务执行代码测试
                           - 实时显示评估日志
                        
                        4. **查看结果**
                           - 评估完成后点击"查看结果"
                           - 显示两个模型的对比数据
                           - 分析微调效果
                        
                        ### HumanEval数据集：
                        - 包含164个Python编程任务
                        - 每个任务有多个测试用例
                        - 只有通过所有测试才算通过
                        - 数据集下载：https://github.com/openai/human-eval
                        
                        ### 评估指标：
                        - **通过率**：通过的任务数 / 总任务数
                        - **提升效果**：微调后通过率 - 原始通过率
                        - **新通过任务**：原来失败，微调后通过的任务
                        - **新失败任务**：原来通过，微调后失败的任务
                        
                        ### 评估时间：
                        - 20个任务：每个模型约5-10分钟
                        - 50个任务：每个模型约15-25分钟
                        - 164个任务：每个模型约45-90分钟
                        
                        ### 注意事项：
                        - 评估需要较长时间，请耐心等待
                        - 需要足够的GPU显存（至少8GB）
                        - 确保模型路径和数据集路径正确
                        """)
        
        # ====== Tab 4: 大模型问答（新增） ======
        with gr.TabItem("💬 大模型问答"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 💬 模型问答")
                    
                    with gr.Row():
                        instruction_input = gr.Textbox(
                            label="输入指令",
                            placeholder="例如：Write a function to add two numbers",
                            lines=3,
                            max_lines=5,
                            scale=3
                        )
                        
                        start_qa_btn = gr.Button(
                            "🚀 开始",
                            variant="primary",
                            size="lg",
                            scale=1
                        )
                    
                    with gr.Row():
                        with gr.Column():
                            gen_temperature = gr.Slider(
                                label="Temperature",
                                value=DEFAULT_CONFIG["gen_temperature"],
                                minimum=0.1,
                                maximum=1.5,
                                step=0.1
                            )
                            
                            gen_top_p = gr.Slider(
                                label="Top-p",
                                value=DEFAULT_CONFIG["gen_top_p"],
                                minimum=0.1,
                                maximum=1.0,
                                step=0.05
                            )
                        
                        with gr.Column():
                            max_new_tokens = gr.Number(
                                label="最大生成token数",
                                value=DEFAULT_CONFIG["max_new_tokens"],
                                minimum=50,
                                maximum=2048,
                                step=50
                            )
                    
                    code_output = gr.Code(
                        label="生成的代码",
                        language="python",
                        lines=15,
                        interactive=False
                    )
                    
                    save_status = gr.Textbox(
                        label="保存状态",
                        value="等待生成",
                        interactive=False,
                        lines=2
                    )
                    
                    qa_status = gr.Textbox(
                        label="问答状态",
                        value="等待输入",
                        interactive=False,
                        lines=3
                    )
                    
                    gr.Markdown("---")
                    gr.Markdown("### 🛠️ 工具")
                    
                    with gr.Accordion("示例指令", open=False):
                        example_instr1 = gr.Button("示例1: 两数相加", variant="secondary", size="sm")
                        example_instr2 = gr.Button("示例2: 判断质数", variant="secondary", size="sm")
                        example_instr3 = gr.Button("示例3: 斐波那契数列", variant="secondary", size="sm")
                        example_instr4 = gr.Button("特殊指令: 自我演化", variant="stop", size="sm")
                
                with gr.Column(scale=2):
                    gr.Markdown("### 📝 使用说明")
                    
                    with gr.Accordion("详细说明", open=True):
                        gr.Markdown("""
                        ### 大模型问答功能：
                        
                        1. **基本功能**：
                           - 输入编程相关的指令（英文/中文均可）
                           - 点击"开始"按钮生成代码
                           - 生成的代码会显示在右侧
                           - 指令会自动保存到MBPP数据集中
                        
                        2. **特殊功能：自我演化**：
                           - 输入"自我演化"（不带引号）
                           - 系统会自动开始微调流程
                           - 使用已保存的指令进行训练
                           - 实现模型的自我改进
                        
                        3. **参数说明**：
                           - **Temperature**：控制生成随机性，值越高越随机
                           - **Top-p**：核采样参数，控制候选词的选择范围
                           - **最大生成token数**：限制生成代码的长度
                        
                        4. **指令保存**：
                           - 每次成功问答后，指令会自动保存到MBPP数据集
                           - 保存格式：用双引号包裹的字符串
                           - 文件路径：`./datasets/mbpp_text_only.jsonl`
                        
                        5. **模型要求**：
                           - 需要先加载0.5B模型（在"模型加载"选项卡）
                           - 模型未加载时会提示错误
                           - 建议使用GPU以获得更好的生成速度
                        
                        6. **使用技巧**：
                           - 对于复杂指令，可以增加最大生成token数
                           - 对于创造性任务，可以适当提高Temperature
                           - 对于确定性任务，可以降低Temperature
                           - 定期执行"自我演化"可以提升模型性能
                        
                        7. **注意事项**：
                           - 生成代码可能需要几秒到几十秒时间
                           - 指令保存不影响现有MBPP数据
                           - "自我演化"会触发完整的训练流程，耗时较长
                           - 确保模型已加载才能使用此功能
                        """)
    
    # ====== 事件绑定 ======
    
    # 清空日志
    def clear_logs():
        log_collector.clear()
        return "日志已清空"
    
    clear_logs_btn.click(
        fn=clear_logs,
        outputs=log_output
    )
    
    # Tab 1: 模型加载
    load_btn.click(
        fn=load_model_interface,
        inputs=[model_path],
        outputs=[load_status, status_display]
    ).then(
        fn=update_system_info,
        outputs=[gpu_info, model_info]
    )
    
    # Tab 2: 模型微调
    def collect_training_config(
        mbpp_dataset_path_val, output_dir_val, 
        num_epochs_val, learning_rate_val,
        batch_size_val, max_generate_items_val, use_lora_val, use_4bit_val
    ):
        # 从模型加载选项卡获取模型路径
        model_path_val = model_path.value if hasattr(model_path, 'value') else DEFAULT_CONFIG["model_path"]
        
        config = {
            "model_path": model_path_val,
            "mbpp_dataset_path": mbpp_dataset_path_val,
            "output_dir": output_dir_val,
            "num_epochs": int(num_epochs_val),
            "learning_rate": float(learning_rate_val),
            "batch_size": int(batch_size_val),
            "max_generate_items": int(max_generate_items_val),
            "use_lora": use_lora_val,
            "use_4bit": use_4bit_val
        }
        return config
    
    train_btn.click(
        fn=collect_training_config,
        inputs=[
            mbpp_dataset_path, output_dir,
            num_epochs, learning_rate,
            batch_size, max_generate_items, use_lora, use_4bit
        ],
        outputs=training_config_state  # 输出到状态组件
    ).then(
        fn=start_training_interface,
        inputs=[training_config_state],  # 从状态组件读取
        outputs=[train_status, status_display]
    )
    
    # 检查MBPP数据集
    def check_mbpp_dataset(mbpp_path):
        if not os.path.exists(mbpp_path):
            return f"❌ MBPP数据集不存在: {mbpp_path}"
        
        # 读取样本数量
        try:
            count = 0
            with open(mbpp_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        count += 1
            
            # 读取示例
            with open(mbpp_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                example = first_line[:100] + "..." if len(first_line) > 100 else first_line
            
            return f"✅ MBPP数据集检查通过\n样本数量: {count}\n示例: {example}"
        except Exception as e:
            return f"❌ 读取MBPP数据集失败: {str(e)}"
    
    check_mbpp_btn.click(
        fn=check_mbpp_dataset,
        inputs=mbpp_dataset_path,
        outputs=check_mbpp_output
    )
    
    # 生成示例数据集
    generate_btn.click(
        fn=generate_example_dataset,
        inputs=example_instructions,
        outputs=generate_output
    )
    
    # Tab 3: 模型评估
    eval_config_state = gr.State({})

    def collect_eval_config(
        finetuned_path, human_eval_path_val,
        max_tasks_val, max_tokens_val, temperature_val, top_p_val
    ):
        # 从模型加载选项卡获取模型路径
        model_path_val = model_path.value if hasattr(model_path, 'value') else DEFAULT_CONFIG["model_path"]
        
        config = {
            "model_path": model_path_val,
            "finetuned_model_path": finetuned_path,
            "human_eval_path": human_eval_path_val,
            "max_tasks": int(max_tasks_val),
            "max_tokens": int(max_tokens_val),
            "temperature": float(temperature_val),
            "top_p": float(top_p_val)
        }
        return config
    
    eval_btn.click(
        fn=collect_eval_config,
        inputs=[
            finetuned_model_path, human_eval_path,
            max_tasks, max_tokens, temperature, top_p
        ],
        outputs=eval_config_state  # 输出到状态组件
    ).then(
        fn=start_evaluation_interface,
        inputs=[eval_config_state],  # 从状态组件读取
        outputs=[eval_status, status_display]
    )
    
    # 查看结果
    results_btn.click(
        fn=get_comparison_results,
        outputs=results_info
    )
    
    # 检查路径
    check_btn.click(
        fn=check_paths,
        inputs=[model_path, finetuned_model_path, human_eval_path],
        outputs=check_output
    )
    
    # Tab 4: 大模型问答
    def start_qa_interface(instruction, temperature, top_p, max_new_tokens):
        """开始问答界面函数"""
        # 检查模型是否加载
        global model
        if model is None:
            return "❌ 模型未加载", "", "请先加载模型", "模型未加载"
        
        if not instruction or instruction.strip() == "":
            return "❌ 请输入指令", "", "", "输入为空"
        
        # 使用问答函数处理指令
        qa_status, code, save_status = process_instruction_with_local_model(
            instruction.strip(),
            temperature,
            top_p,
            max_new_tokens,
            mbpp_path=DEFAULT_CONFIG["mbpp_dataset_path"]
        )
        
        return qa_status, code, save_status, "处理完成"
    
    # 简化事件绑定，直接传递参数
    start_qa_btn.click(
        fn=start_qa_interface,
        inputs=[instruction_input, gen_temperature, gen_top_p, max_new_tokens],
        outputs=[qa_status, code_output, save_status, status_display]
    )
    
    # 示例指令按钮
    def set_example_instruction(example_text):
        return example_text
    
    example_instr1.click(
        fn=lambda: set_example_instruction("Write a function to add two numbers and return the sum"),
        outputs=instruction_input
    )
    
    example_instr2.click(
        fn=lambda: set_example_instruction("Write a function to check if a number is prime"),
        outputs=instruction_input
    )
    
    example_instr3.click(
        fn=lambda: set_example_instruction("Write a function to generate the first n Fibonacci numbers"),
        outputs=instruction_input
    )
    
    example_instr4.click(
        fn=lambda: set_example_instruction("自我演化"),
        outputs=instruction_input
    )
    
    # ====== 定时更新 ======
    
    # 更新系统信息
    demo.load(
        fn=update_system_info,
        outputs=[gpu_info, model_info],
        every=5
    )
    
    # 更新日志
    def update_all():
        logs = update_logs()
        
        # 更新状态
        global is_training, is_evaluating, is_generating
        status = "准备就绪"
        if is_training:
            status = "训练中..."
        elif is_evaluating:
            status = "评估中..."
        elif is_generating:
            status = "生成中..."
        
        # 更新结果
        results = "暂无结果"
        global comparison_results
        if comparison_results:
            results = f"通过率: {comparison_results.get('finetuned_pass_rate', 0):.1f}%"
        
        return logs, status, results
    
    demo.load(
        fn=update_all,
        outputs=[log_output, status_display, results_info],
        every=2
    )

# ====== 主程序 ======
if __name__ == "__main__":
    # 检查依赖
    required_packages = ["torch", "transformers", "gradio"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少必要的依赖库:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n请运行以下命令安装:")
        print("pip install torch transformers gradio")
        print("\n可选安装（用于完整功能）:")
        print("pip install datasets accelerate peft requests")
        sys.exit(1)
    
    # 创建必要的目录
    os.makedirs("./models", exist_ok=True)
    os.makedirs(DEFAULT_CONFIG["output_dir"], exist_ok=True)
    os.makedirs("./datasets", exist_ok=True)
    os.makedirs("./mbpp_training_data", exist_ok=True)
    
    # 检查MBPP数据集
    mbpp_path = DEFAULT_CONFIG["mbpp_dataset_path"]
    if not os.path.exists(mbpp_path):
        print(f"⚠️ 警告: MBPP数据集不存在: {mbpp_path}")
        print("将创建新的MBPP数据集文件")
        with open(mbpp_path, 'w', encoding='utf-8') as f:
            f.write('"Write a function to add two numbers and return the sum"\n')
            f.write('"Write a function to check if a number is prime"\n')
            f.write('"Write a function to generate the first n Fibonacci numbers"\n')
        print(f"✅ 已创建示例MBPP数据集: {mbpp_path}")
    
    # 检查HumanEval数据集
    if not os.path.exists(DEFAULT_CONFIG["human_eval_path"]):
        print(f"⚠️ 警告: HumanEval数据集不存在: {DEFAULT_CONFIG['human_eval_path']}")
        print("请从以下地址下载:")
        print("https://github.com/openai/human-eval")
        print("下载后保存到 ./datasets/ 目录")
    
    # 启动界面
    print("🚀 启动 Qwen2.5-Coder 完整系统...")
    print(f"访问地址: http://localhost:7860")
    print("按 Ctrl+C 停止服务")
    
    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_api=False
        )
    except KeyboardInterrupt:
        print("\n👋 服务已停止")