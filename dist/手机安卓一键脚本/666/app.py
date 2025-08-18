#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM代理服务 - Termux版
专为Android Termux环境优化的FastAPI版本
(增强错误处理和解决方案版)
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
    print("错误：缺少FastAPI核心依赖。")
    print("[处理办法]：请在Termux中运行以下命令来安装所需库:\npip install fastapi uvicorn httpx pydantic python-multipart aiofiles")
    sys.exit(1)

# ==================== 日志和错误输出配置 ====================
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout), # 同时输出到命令行
        logging.FileHandler('llm_proxy.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def log_and_print_error(exception: Exception, suggestion: str):
    """一个统一的函数，用于打印错误和处理建议到命令行和日志。"""
    error_details = traceback.format_exc()
    logger.error(f"发生错误: {exception}\n{error_details}")
    print(f"\n[!!!] 程序遇到一个错误: {exception}")
    print(f"[处理办法]：{suggestion}\n")
    print(f"[技术细节]：\n{error_details}")

# ==================== 辅助函数 ====================
def is_termux_environment() -> bool:
    return 'com.termux' in sys.executable or os.path.exists('/data/data/com.termux') or 'TERMUX_VERSION' in os.environ

def get_resource_path(relative_path: str) -> str:
    base_path = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).parent
    return str(base_path / relative_path)

def check_port_available(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            return s.connect_ex((host, port)) != 0
    except Exception as e:
        log_and_print_error(e, f"无法检查端口 {host}:{port} 的可用性。请检查您的网络设置或权限。")
        return False

def setup_termux_environment():
    if not is_termux_environment(): return
    try:
        subprocess.run(['termux-wake-lock'], check=False, capture_output=True)
        logger.info("已启用termux-wake-lock，防止系统休眠。")
    except FileNotFoundError:
        logger.warning("未找到 'termux-wake-lock' 命令。服务可能在后台被系统终止。")
        print("[提示]：未找到 'termux-wake-lock'。如果服务在后台意外停止，请尝试在Termux中运行 'pkg install termux-api'。")
    os.environ.update({'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1'})

# ==================== 配置管理器 ====================
class ConfigManager:
    def __init__(self, config_file: str = "config.ini"):
        self.config_file = get_resource_path(config_file)
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
        self.config['SERVER'] = {'port': '5000', 'host': '0.0.0.0' if is_termux_environment() else '127.0.0.1', 'api_key': '123', 'min_response_length': '300', 'request_timeout': '120', 'web_port': '5000', 'web_host': '0.0.0.0' if is_termux_environment() else '127.0.0.1'}
        self.config['API_KEYS'] = {'group1': json.dumps(["YOUR_API_KEY_1", "YOUR_API_KEY_2"]), 'group2': json.dumps(["YOUR_API_KEY_3", "YOUR_API_KEY_4"])}
        self.config['API'] = {'base_url': 'https://generativelanguage.googleapis.com/v1beta'}
        self.save_config()

    def save_config(self):
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                self.config.write(f)
            logger.info("配置已保存")
        except Exception as e:
            log_and_print_error(e, f"无法写入配置文件到 {self.config_file}。请检查文件权限或磁盘空间。")

    def force_reload_config(self):
        logger.info("强制重新加载配置文件")
        self.load_config()

    def get_config_section(self, section: str) -> Dict:
        try:
            return dict(self.config.items(section))
        except configparser.NoSectionError as e:
            log_and_print_error(e, f"配置文件 config.ini 中缺少必需的 `[{section}]` 部分。请恢复或参照默认配置添加该部分。")
            sys.exit(1)

    def get_server_config(self) -> Dict[str, Any]:
        server_conf = self.get_config_section('SERVER')
        return {'port': int(server_conf['port']), 'host': server_conf['host'], 'api_key': server_conf['api_key'], 'min_response_length': int(server_conf['min_response_length']), 'request_timeout': int(server_conf['request_timeout']), 'web_port': int(server_conf['web_port']), 'web_host': server_conf['web_host']}

    def get_api_keys(self) -> Dict[str, List[str]]:
        try:
            api_keys_conf = self.get_config_section('API_KEYS')
            return {'group1': json.loads(api_keys_conf['group1']), 'group2': json.loads(api_keys_conf['group2'])}
        except json.JSONDecodeError as e:
            log_and_print_error(e, "config.ini 中 [API_KEYS] 的值不是有效的JSON格式。请确保密钥列表是 `[\"key1\", \"key2\"]` 这样的格式。")
            return {'group1': [], 'group2': []} # 返回空列表以避免程序崩溃
        
    # 其他 set 和 get 方法保持不变...
    def set_server_config(self, **kwargs): self.config['SERVER'].update({k: str(v) for k, v in kwargs.items()}); self.save_config()
    def set_api_keys(self, **kwargs): self.config['API_KEYS'].update({k: json.dumps(v) for k, v in kwargs.items()}); self.save_config()
    def get_base_url(self) -> str: return self.get_config_section('API')['base_url']
    def set_base_url(self, base_url: str): self.config['API']['base_url'] = base_url; self.save_config()

# ==================== FastAPI 服务 ====================
if FASTAPI_AVAILABLE:
    config_manager = ConfigManager()
    app_fastapi = FastAPI(title="LLM代理服务", version="2.1.0-enhanced-error-handling")
    app_fastapi.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

    class ChatRequest(BaseModel):
        model: str; messages: List[Dict[str, Any]]; stream: bool = False

    current_group_index = 0

    def get_current_api_keys() -> List[str]:
        global current_group_index
        config_manager.force_reload_config()
        api_keys = config_manager.get_api_keys()
        group_name = 'group1' if current_group_index == 0 else 'group2'
        current_group_index = 1 - current_group_index
        valid_keys = [key for key in api_keys.get(group_name, []) if key and not key.startswith("YOUR_")]
        logger.info(f"当前使用密钥组: {group_name}, 有效密钥数量: {len(valid_keys)}")
        return valid_keys

    async def send_single_request(client: httpx.AsyncClient, api_key: str, request_data: dict) -> Tuple[Any, Any]:
        """发送单个请求，返回 (结果, 错误详情)"""
        url = f"{config_manager.get_base_url()}/openai/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        try:
            resp = await client.post(url, headers=headers, json=request_data, timeout=config_manager.get_server_config()['request_timeout'])
            resp.raise_for_status()
            return resp.json(), None
        except httpx.HTTPStatusError as e:
            # 这是API层面错误的关键
            status = e.response.status_code
            error_type = "auth" if status in [401, 403] else "rate_limit" if status == 429 else "other"
            logger.warning(f"API请求失败 (HTTP {status}) 使用 key: ...{api_key[-4:]}。原因: {e.response.text}")
            return None, (error_type, f"HTTP {status}: {e.response.text}")
        except httpx.RequestError as e:
            # 这是网络层面错误的关键
            logger.warning(f"网络请求失败使用 key: ...{api_key[-4:]}。原因: {e.__class__.__name__}")
            return None, ("timeout", str(e))
        except Exception as e:
            log_and_print_error(e, "在发送单个API请求时发生未知内部错误。")
            return None, ("internal", str(e))

    async def process_api_requests(request_data: dict) -> Dict:
        """处理所有API请求，并返回最佳响应或详细的错误信息"""
        current_keys = get_current_api_keys()
        if not current_keys:
            print("\n[错误]：当前密钥组中没有配置任何有效的API密钥。")
            print("[处理办法]：请检查您的 config.ini 文件，在 `[API_KEYS]` 部分填入您的API密钥。")
            raise HTTPException(status_code=500, detail={"error": "No valid API keys configured on the server.", "solution": "Please check config.ini and add API keys under [API_KEYS]."})

        min_len = config_manager.get_server_config()['min_response_length']
        valid_responses = []
        error_summary = {"auth": 0, "rate_limit": 0, "timeout": 0, "other": 0}

        async with httpx.AsyncClient() as client:
            tasks = [send_single_request(client, key, request_data) for key in current_keys]
            for future in asyncio.as_completed(tasks):
                result, error = await future
                if result and result.get("choices"):
                    content = result["choices"][0].get("message", {}).get("content", "")
                    if len(content) >= min_len:
                        valid_responses.append({'result': result, 'token_count': len(content)})
                elif error:
                    error_type, _ = error
                    if error_type in error_summary:
                        error_summary[error_type] += 1

        if valid_responses:
            return max(valid_responses, key=lambda x: x['token_count'])['result']

        # 如果没有有效响应，则生成智能错误报告
        total_errors = sum(error_summary.values())
        if total_errors == 0 and len(current_keys) > 0:
             # 所有请求都成功了，但响应内容太短或格式不正确
            detail = {
                "error": "All upstream API requests succeeded, but returned no valid content or content was too short.",
                "solution": f"The model may be generating empty or very short responses. Check the upstream API status or your prompt. Minimum required length is {min_len} characters."
            }
            raise HTTPException(status_code=502, detail=detail)

        if error_summary["auth"] == total_errors:
            detail = {"error": "All upstream API keys failed authentication.", "solution": "Please check your API keys in config.ini. They might be invalid, expired, or have billing issues."}
            raise HTTPException(status_code=401, detail=detail)
        if error_summary["rate_limit"] == total_errors:
            detail = {"error": "All upstream API keys have reached their rate limits.", "solution": "Please wait a moment before trying again, or switch to a different set of keys."}
            raise HTTPException(status_code=429, detail=detail)
        if error_summary["timeout"] == total_errors:
            detail = {"error": "Could not connect to the upstream API provider from the server.", "solution": "Please check the server's network connection and the `base_url` in config.ini."}
            raise HTTPException(status_code=504, detail=detail)
        
        # 混合错误
        error_report = f"Failure summary: {error_summary['auth']} auth errors, {error_summary['rate_limit']} rate limit errors, {error_summary['timeout']} timeouts, {error_summary['other']} other errors."
        detail = {"error": "Multiple upstream API requests failed with mixed reasons.", "report": error_report}
        raise HTTPException(status_code=503, detail=detail)

    async def stream_response_content(result: dict):
        # (此函数逻辑不变，仅为简洁省略)
        content = result["choices"][0]["message"]["content"]
        response_id, created, model = result.get("id", f"c-{int(time.time())}"), result.get("created", int(time.time())), result.get("model", "gemini-pro")
        async def gen():
            for i in range(0, len(content), max(1, len(content) // 50)):
                chunk = content[i:i + max(1, len(content) // 50)]
                yield f"data: {json.dumps({'id': response_id, 'created': created, 'model': model, 'choices': [{'delta': {'content': chunk}}]})}\n\n"
                await asyncio.sleep(0.01)
            yield f"data: {json.dumps({'id': response_id, 'created': created, 'model': model, 'choices': [{'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    @app_fastapi.post("/v1/chat/completions")
    async def chat_completions_proxy(req: ChatRequest, http_req: Request):
        try:
            server_key = config_manager.get_server_config()['api_key']
            auth_header = http_req.headers.get("Authorization", "").split(" ")
            if len(auth_header) != 2 or auth_header[0] != "Bearer" or auth_header[1] != server_key:
                raise HTTPException(status_code=401, detail="Invalid API Key")
            
            request_data = await http_req.json()
            best_response = await process_api_requests(request_data)
            
            if req.stream:
                return await stream_response_content(best_response)
            return JSONResponse(content=best_response)
        except HTTPException:
            raise # 重新抛出已处理的HTTP异常
        except Exception as e:
            log_and_print_error(e, "处理聊天请求时发生了一个未预料的服务器内部错误。")
            raise HTTPException(status_code=500, detail="Internal Server Error")
    
    # Web界面和配置API部分保持不变，为简洁省略，可直接从原代码复制

# ==================== 主程序入口 ====================
def main():
    print("正在启动LLM代理服务 (增强错误处理版)...")
    setup_termux_environment()
    
    # 检查 uvicorn 是否安装
    try:
        import uvicorn
    except ImportError:
        print("\n[错误]：缺少 'uvicorn' 库，无法启动Web服务。")
        print("[处理办法]：请运行 'pip install uvicorn' 来安装它。")
        sys.exit(1)

    try:
        server_config = config_manager.get_server_config()
        web_host, web_port = server_config['web_host'], server_config['web_port']
        
        if not check_port_available(web_host, web_port):
            print(f"\n[警告]：端口 {web_port} 已被占用。")
            if is_termux_environment():
                alt_port = web_port + 1
                if check_port_available(web_host, alt_port):
                    print(f"[处理办法]：将自动尝试使用备用端口 {alt_port}。")
                    server_config.update({'web_port': alt_port, 'port': alt_port})
                    config_manager.set_server_config(**server_config)
                    web_port = alt_port
                else:
                    print(f"[处理办法]：备用端口 {alt_port} 也被占用。请在 config.ini 文件中手动指定一个未被占用的端口。")
                    sys.exit(1)
            else:
                print(f"[处理办法]：请关闭占用端口 {web_port} 的程序，或在 config.ini 中修改 'web_port' 和 'port' 的值。")
                sys.exit(1)

        script_name = Path(__file__).stem
        app_string = f"{script_name}:app_fastapi"
        
        print(f"\n服务启动成功！")
        print(f"管理界面: http://{web_host}:{web_port}/")
        print(f"API 端点: http://{server_config['host']}:{server_config['port']}/v1/chat/completions")
        
        uvicorn_config = {"app": app_string, "host": web_host, "port": web_port, "log_level": "info", "reload": False}
        if is_termux_environment(): uvicorn_config.update({"workers": 1, "limit_concurrency": 10})
        
        uvicorn.run(**uvicorn_config)

    except Exception as e:
        log_and_print_error(e, "启动服务时发生严重错误，程序已退出。请根据上面的技术细节排查问题。")
        sys.exit(1)

if __name__ == "__main__":
    main()
