"""
模型训练模块
"""
import os
import json
import threading
import time
import tempfile
import traceback
from datetime import datetime
from typing import Tuple
from ..config.settings import DEFAULT_CONFIG, API_CONFIG
from ..utils import log, process_single_instruction

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
            import torch
            from transformers import (
                AutoTokenizer,
                AutoModelForCausalLM,
                TrainingArguments,
                Trainer,
                DataCollatorForLanguageModeling
            )
            
            from datasets import Dataset
            import warnings
            warnings.filterwarnings("ignore")
            
            self.log("库导入完成")
            
            # 加载模型
            self.log(f"加载模型: {self.config['model_path']}")
            
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
                fp16=False,
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
            self.log(traceback.format_exc())

def start_training_interface(config_data):
    """开始训练界面函数"""
    global is_training, training_thread
    is_training = False
    training_thread = None
    
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

is_training = False
training_thread = None
