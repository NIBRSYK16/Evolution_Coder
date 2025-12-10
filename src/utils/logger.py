"""
日志收集和管理模块
"""
import threading
from datetime import datetime

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
