"""
Qwen2.5-Coder 完整系统 - src包初始化
"""

__version__ = "1.0.0"
__author__ = "Evolution Coder"

# 延迟导入以避免循环导入
def __getattr__(name):
    """懒加载模块"""
    if name == 'DEFAULT_CONFIG':
        from .config.settings import DEFAULT_CONFIG
        return DEFAULT_CONFIG
    elif name == 'API_CONFIG':
        from .config.settings import API_CONFIG
        return API_CONFIG
    elif name == 'log':
        from .utils import log
        return log
    elif name == 'log_collector':
        from .utils import log_collector
        return log_collector
    elif name == 'load_model_interface':
        from .models import load_model_interface
        return load_model_interface
    elif name == 'get_model':
        from .models import get_model
        return get_model
    elif name == 'is_model_loaded':
        from .models import is_model_loaded
        return is_model_loaded
    elif name == 'start_training_interface':
        from .training import start_training_interface
        return start_training_interface
    elif name == 'start_evaluation_interface':
        from .evaluation import start_evaluation_interface
        return start_evaluation_interface
    elif name == 'get_comparison_results':
        from .evaluation import get_comparison_results
        return get_comparison_results
    elif name == 'create_interface':
        from .ui import create_interface
        return create_interface
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'DEFAULT_CONFIG',
    'API_CONFIG',
    'log',
    'log_collector',
    'load_model_interface',
    'get_model',
    'is_model_loaded',
    'start_training_interface',
    'start_evaluation_interface',
    'get_comparison_results',
    'create_interface'
]
