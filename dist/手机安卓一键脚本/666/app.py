#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM代理服务 - Termux版
专为Android Termux环境优化的FastAPI版本
(最终修复版：包含增强报错和完整的Web UI，全中文提示)
"""

import asyncio
import httpx
import os
import sys
import json
import time
import logging
import configparser
import socket
import subprocess
import traceback
from pathlib import Path
from typing import List, Dict, Any, Tuple

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
    print("错误：缺少运行本服务所需的核心库。")
    print("[处理办法]：请在Termux中运行以下命令来安装所需库:\npip install fastapi uvicorn httpx pydantic python-multipart aiofiles jinja2")
    sys.exit(1)

# ==================== 日志和错误输出配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('llm_proxy.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def log_and_print_error(exception: Exception, suggestion: str):
    """一个统一的函数，用于在命令行和日志中打印中文错误和处理建议。"""
    error_details = traceback.format_exc()
    logger.error(f"发生错误: {exception}\n{error_details}")
    print(f"\n[!!!] 程序遇到一个错误: {exception}")
    print(f"[处理办法]：{suggestion}\n")
    print(f"[技术细节]：\n{error_details}")

# ==================== 辅助函数 ====================
def is_termux_environment() -> bool:
    return 'com.termux' in sys.executable or os.path.exists('/data/data/com.termux') or 'TERMUX_VERSION' in os.environ

def get_resource_path(relative_path: str) -> Path:
    base_path = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).parent
    return base_path / relative_path

def check_port_available(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            return s.connect_ex((host, port)) != 0
    except Exception as e:
        log_and_print_error(e, f"无法检查端口 {host}:{port} 的可用性。请检查您的网络设置或Termux权限。")
        return False

def setup_termux_environment():
    if not is_termux_environment(): return
    try:
        subprocess.run(['termux-wake-lock'], check=False, capture_output=True)
        logger.info("已启用termux-wake-lock，防止系统休眠。")
    except FileNotFoundError:
        logger.warning("未找到 'termux-wake-lock' 命令。服务可能在后台被系统终止。")
        print("[提示]：未找到 'termux-wake-lock'。如果服务在后台意外停止，可尝试在Termux中运行 'pkg install termux-api'。")
    os.environ.update({'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1'})

# ==================== 配置管理器 ====================
class ConfigManager:
    def __init__(self, config_file: str = "config.ini"):
        self.config_file = str(get_resource_path(config_file))
        self.config = configparser.ConfigParser()
        self.load_config()

    def load_config(self):
        try:
            if not os.path.exists(self.config_file):
                logger.warning(f"配置文件 {self.config_file} 不存在，将创建默认配置。")
                print(f"提示：未找到配置文件，已在当前目录创建默认的 config.ini 文件。")
                self.create_default_config()
            self.config.clear()
            self.config.read(self.config_file, encoding='utf-8')
        except configparser.Error as e:
            log_and_print_error(e, "请检查 config.ini 文件的格式是否正确。例如，确保每个部分都有 `[SECTION_NAME]` 这样的标题，并且键值对使用 `=` 分隔。")
            sys.exit(1)

    def create_default_config(self):
        host = '0.0.0.0' if is_termux_environment() else '127.0.0.1'
        self.config['SERVER'] = {'port': '5000', 'host': host, 'api_key': '123', 'min_response_length': '300', 'request_timeout': '120', 'web_port': '5000', 'web_host': host}
        self.config['API_KEYS'] = {'group1': json.dumps(["请在此处填入您的API密钥1", "请在此处填入您的API密钥2"]), 'group2': json.dumps(["请在此处填入您的API密钥3", "请在此处填入您的API密钥4"])}
        self.config['API'] = {'base_url': 'https://generativelanguage.googleapis.com/v1beta'}
        self.save_config()

    def save_config(self):
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                self.config.write(f)
            logger.info("配置已保存")
        except Exception as e:
            log_and_print_error(e, f"无法写入配置文件到 {self.config_file}。请检查文件权限或磁盘空间。")

    def force_reload_config(self): self.load_config()
    def get_config_section(self, section: str): return dict(self.config.items(section))
    def get_server_config(self): conf = self.get_config_section('SERVER'); return {k: int(v) if k in ['port', 'min_response_length', 'request_timeout', 'web_port'] else v for k, v in conf.items()}
    def get_api_keys(self):
        try:
            conf = self.get_config_section('API_KEYS')
            return {k: json.loads(v) for k, v in conf.items()}
        except json.JSONDecodeError as e:
            log_and_print_error(e, "config.ini 中 [API_KEYS] 的值不是有效的JSON格式。请确保密钥列表是 `[\"密钥1\", \"密钥2\"]` 这样的格式。")
            return {'group1': [], 'group2': []}
    def get_base_url(self): return self.get_config_section('API')['base_url']
    def set_server_config(self, **kwargs): self.config['SERVER'].update({k: str(v) for k, v in kwargs.items()}); self.save_config()
    def set_api_keys(self, **kwargs): self.config['API_KEYS'].update({k: json.dumps(v) for k, v in kwargs.items()}); self.save_config()
    def set_base_url(self, url: str): self.config['API']['base_url'] = url; self.save_config()


# ==================== FastAPI 应用初始化 ====================
if FASTAPI_AVAILABLE:
    config_manager = ConfigManager()
    app_fastapi = FastAPI(title="LLM代理服务", version="2.2-fixed-zh")
    app_fastapi.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    
    class ChatRequest(BaseModel): model: str; messages: List[Dict[str, Any]]; stream: bool = False
    current_group_index = 0

    # ==================== API 核心逻辑 ====================
    def get_current_api_keys() -> List[str]:
        global current_group_index; config_manager.force_reload_config(); api_keys = config_manager.get_api_keys()
        group = 'group1' if current_group_index == 0 else 'group2'; current_group_index = 1 - current_group_index
        valid = [k for k in api_keys.get(group, []) if k and not k.startswith("请在此处")]; logger.info(f"使用密钥组: {group}, 有效密钥数: {len(valid)}"); return valid

    async def send_single_request(client: httpx.AsyncClient, api_key: str, data: dict) -> Tuple[Any, Any]:
        url = f"{config_manager.get_base_url()}/openai/chat/completions"; headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        try: resp = await client.post(url, headers=headers, json=data, timeout=config_manager.get_server_config()['request_timeout']); resp.raise_for_status(); return resp.json(), None
        except httpx.HTTPStatusError as e: status = e.response.status_code; error_type = "auth" if status in [401, 403] else "rate_limit" if status == 429 else "other"; logger.warning(f"API请求失败 (HTTP {status}) 使用 key: ...{api_key[-4:]}。"); return None, (error_type, f"HTTP {status}")
        except httpx.RequestError as e: logger.warning(f"网络请求失败使用 key: ...{api_key[-4:]}。原因: {e.__class__.__name__}"); return None, ("timeout", str(e))
        except Exception as e: log_and_print_error(e, "发送单个API请求时发生未知内部错误。"); return None, ("internal", str(e))

    async def process_api_requests(request_data: dict) -> Dict:
        keys = get_current_api_keys()
        if not keys: print("[处理办法]：请检查 config.ini 文件，在 `[API_KEYS]` 部分填入您有效的API密钥。"); raise HTTPException(500, {"error": "服务器未配置任何有效的API密钥。"})
        
        min_len, valid_responses, errors = config_manager.get_server_config()['min_response_length'], [], {"auth": 0, "rate_limit": 0, "timeout": 0, "other": 0}
        async with httpx.AsyncClient() as client:
            tasks = [send_single_request(client, key, request_data) for key in keys]
            for future in asyncio.as_completed(tasks):
                result, error = await future
                if result and result.get("choices"): content = result["choices"][0].get("message", {}).get("content", ""); valid_responses.append({'result': result, 'token_count': len(content)}) if len(content) >= min_len else None
                elif error: error_type, _ = error; errors[error_type] += 1 if error_type in errors else 0
        
        if valid_responses: return max(valid_responses, key=lambda x: x['token_count'])['result']
        
        total_err = sum(
