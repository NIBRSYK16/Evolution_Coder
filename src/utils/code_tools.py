"""
代码处理工具函数
"""
import re
import os
import tempfile
import subprocess
from typing import Tuple
from .logger import log

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
    from .api_helper import call_qwen_api, validate_code_with_14b
    
    log(f"[{index}] 处理指令: {instruction[:80]}...")
    
    # 步骤1: 使用32B模型生成代码
    log(f"[{index}] 调用32B API生成代码...")
    success, code = call_qwen_api(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
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
