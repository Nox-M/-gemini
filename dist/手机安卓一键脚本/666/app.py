#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM代理服务 - Termux版
专为Android Termux环境优化的FastAPI版本
【已修改以提供详细错误报告】
"""

import asyncio
import httpx
import os
import sys
import json
import time
import logging
import configparser
import signal
import platform
import socket
import subprocess
from pathlib import Path
from typing import List, Dict, Any

# FastAPI和Pydantic导入
try:
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("错误：缺少FastAPI依赖。请运行 'pip install fastapi uvicorn httpx pydantic python-multipart aiofiles'")
    sys.exit(1)

# ==================== 辅助函数 ====================
def is_termux_environment() -> bool:
    """检测是否在Termux环境中运行"""
    return (
        'com.termux' in sys.executable or
        os.path.exists('/data/data/com.termux') or
        'TERMUX_VERSION' in os.environ or
        os.path.exists('/system/bin/termux-setup-storage')
    )

def get_resource_path(relative_path: str) -> str:
    """获取资源的绝对路径，兼容开发环境和PyInstaller打包环境"""
    if getattr(sys, 'frozen', False):
        # 如果是打包后的exe文件
        base_path = Path(sys.executable).parent
    else:
        # 如果是正常的python脚本
        base_path = Path(__file__).parent
    return str(base_path / relative_path)

def check_port_available(host: str, port: int) -> bool:
    """检查端口是否可用"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex((host, port))
            return result != 0
    except Exception:
        return False

def setup_termux_environment():
    """设置Termux环境优化"""
    if not is_termux_environment():
        return
    
    try:
        # 检查并设置termux-wake-lock以保持服务运行
        subprocess.run(['termux-wake-lock'], check=False, capture_output=True)
        logger.info("已启用termux-wake-lock")
    except FileNotFoundError:
        logger.warning("termux-wake-lock未找到，服务可能会在后台被系统终止")
    
    # 设置环境变量以优化Termux性能
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUNBUFFERED'] = '1'

# ==================== 配置管理器 ====================
class ConfigManager:
    """配置文件管理器"""
    
    def __init__(self, config_file: str = "config.ini"):
        self.config_file = get_resource_path(config_file)
        self.config = configparser.ConfigParser()
        self.load_config()
    
    def load_config(self):
        """加载配置文件"""
        # 清除现有配置
        self.config.clear()
        
        if os.path.exists(self.config_file):
            # 重新读取配置文件，确保获取最新内容
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config.read_file(f)
        else:
            self.create_default_config()
    
    def create_default_config(self):
        """创建默认配置"""
        # 根据环境设置默认配置
        if is_termux_environment():
            default_port = '5000'
            default_host = '0.0.0.0'
            default_timeout = '120'  # Termux环境下使用较短的超时时间
            default_min_length = '300'  # Termux环境下使用较小的最小响应长度
            default_web_port = '5000'  # Web界面使用不同端口
        else:
            default_port = '5000'
            default_host = '127.0.0.1'
            default_timeout = '180'
            default_min_length = '400'
            default_web_port = '5000'  # Web界面使用不同端口
        
        self.config['SERVER'] = {
            'port': default_port,
            'host': default_host,
            'api_key': '123',
            'min_response_length': default_min_length,
            'request_timeout': default_timeout,
            'web_port': default_web_port,
            'web_host': default_host
        }
        
        self.config['API_KEYS'] = {
            'group1': json.dumps([
                "AIzaSyCgh-9h5PhprwiGSrk7oNxD5Bl240gI6Fk",
                "AIzaSyBmfY6uDjeDmaCbjjuDpMhLJe6H8nMMGXA",
                "AIzaSyCRxaB09p2wEDJPbwc69tEukfrsv0HT5YQ",
                "AIzaSyDJqNc2s-L2_RW0-AwMevHRvhYgEMMXLRM"
            ]),
            'group2': json.dumps([
                "AIzaSyDxG_Dn27XZ-OSeg_iWbGduohqD9gYrGiI",
                "AIzaSyDP-WGwWX4SY2uLTaKAivWwuXzX0LqSui0",
                "AIzaSyBwlIzbZ7bnRtYU7iicNdMnLYKkd8XVPDU",
                "AIzaSyDIwwW4ApVM7Dsj7BuCq4766eCWcOW9_mM"
            ])
        }
        
        self.config['API'] = {
            'base_url': 'https://generativelanguage.googleapis.com/v1beta'
        }
        
        self.save_config()
    
    def save_config(self):
        """保存配置到文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                self.config.write(f)
            logger.info("配置已保存")
            # 保存后立即重新加载配置，确保内存中的配置是最新的
            self.load_config()
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            raise
    
    def force_reload_config(self):
        """强制重新加载配置文件"""
        logger.info("强制重新加载配置文件")
        self.load_config()
    
    def get_server_config(self) -> Dict[str, Any]:
        """获取服务器配置"""
        return {
            'port': int(self.config['SERVER']['port']),
            'host': self.config['SERVER']['host'],
            'api_key': self.config['SERVER']['api_key'],
            'min_response_length': int(self.config['SERVER']['min_response_length']),
            'request_timeout': int(self.config['SERVER']['request_timeout']),
            'web_port': int(self.config['SERVER']['web_port']),
            'web_host': self.config['SERVER']['web_host']
        }
    
    def set_server_config(self, port: int, host: str, api_key: str, 
                         min_response_length: int, request_timeout: int,
                         web_port: int, web_host: str):
        """设置服务器配置"""
        self.config['SERVER']['port'] = str(port)
        self.config['SERVER']['host'] = host
        self.config['SERVER']['api_key'] = api_key
        self.config['SERVER']['min_response_length'] = str(min_response_length)
        self.config['SERVER']['request_timeout'] = str(request_timeout)
        self.config['SERVER']['web_port'] = str(web_port)
        self.config['SERVER']['web_host'] = web_host
        self.save_config()
    
    def get_api_keys(self) -> Dict[str, List[str]]:
        """获取API密钥"""
        return {
            'group1': json.loads(self.config['API_KEYS']['group1']),
            'group2': json.loads(self.config['API_KEYS']['group2'])
        }
    
    def set_api_keys(self, group1: List[str], group2: List[str]):
        """设置API密钥"""
        self.config['API_KEYS']['group1'] = json.dumps(group1)
        self.config['API_KEYS']['group2'] = json.dumps(group2)
        self.save_config()
    
    def get_base_url(self) -> str:
        """获取基础URL"""
        return self.config['API']['base_url']
    
    def set_base_url(self, base_url: str):
        """设置基础URL"""
        self.config['API']['base_url'] = base_url
        self.save_config()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('llm_proxy.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# 全局配置管理器实例
config_manager = ConfigManager()

# ==================== FastAPI服务 (如果可用) ====================
if FASTAPI_AVAILABLE:
    class ChatRequest(BaseModel):
        model: str
        messages: List[Dict[str, Any]]
        temperature: float = 0.7
        max_tokens: int = 4096
        stream: bool = False

    # 轮询计数器
    current_group_index = 0

    def get_current_api_keys():
        """根据轮询机制返回当前应该使用的API密钥组"""
        global current_group_index
        
        # 每次获取密钥前都强制重新加载配置，确保获取最新密钥
        config_manager.force_reload_config()
        api_keys = config_manager.get_api_keys()
        
        if current_group_index == 0:
            keys = api_keys['group1']
            current_group_index = 1
        else:
            keys = api_keys['group2']
            current_group_index = 0
        
        valid_keys = [key for key in keys if key and not key.startswith("YOUR_") and len(key) > 10]
        
        # 记录当前使用的密钥组信息，便于调试
        logger.info(f"当前使用密钥组: {'group1' if current_group_index == 1 else 'group2'}, 有效密钥数量: {len(valid_keys)}")
        
        return valid_keys

    # ▼▼▼▼▼ 第一个修改点 ▼▼▼▼▼
    async def send_single_request(client: httpx.AsyncClient, api_key: str, request_data: dict):
        """
        【修改版】使用单个API密钥发送请求。
        失败时会直接抛出 httpx.HTTPStatusError 或 httpx.RequestError 异常。
        """
        cleaned_data = {}
        supported_params = {
            'model', 'messages', 'temperature', 'max_tokens',
            'top_p', 'top_k', 'stop'
        }
        
        for key, value in request_data.items():
            if key in supported_params:
                cleaned_data[key] = value
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        url = f"{config_manager.get_base_url()}/openai/chat/completions"
        
        # 不再使用 try-except 包裹，让异常直接抛出给调用者处理
        response = await client.post(url, headers=headers, json=cleaned_data,
                                   timeout=config_manager.get_server_config()['request_timeout'])
        
        # 如果响应状态码是 4xx 或 5xx，这行会自动抛出 httpx.HTTPStatusError 异常
        response.raise_for_status()
        
        # 如果请求成功，直接返回原始的 httpx.Response 对象
        return response
    # ▲▲▲▲▲ 第一个修改点结束 ▲▲▲▲▲

    async def stream_response_content(result: dict, content: str):
        """将完整的响应内容以流式方式发送给前端"""
        response_id = result.get("id", f"chatcmpl-{int(time.time())}")
        created_time = result.get("created", int(time.time()))
        model_name = result.get("model", "gemini-2.5
