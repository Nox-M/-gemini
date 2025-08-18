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
        
        total_err = sum(errors.values())
        if total_err == 0 and keys: raise HTTPException(502, {"error": "所有API请求均成功，但未返回有效内容或内容过短。", "solution": f"模型可能生成了空回复。最小要求长度为 {min_len} 字符。"})
        if errors["auth"] == total_err: raise HTTPException(401, {"error": "所有上游API密钥均认证失败。", "solution": "请检查您在config.ini中的API密钥是否正确、有效或已绑定支付方式。"})
        if errors["rate_limit"] == total_err: raise HTTPException(429, {"error": "所有上游API密钥均达到速率限制。", "solution": "请稍后再试，或更换一批API密钥。"})
        if errors["timeout"] == total_err: raise HTTPException(504, {"error": "无法连接到上游API供应商。", "solution": "请检查服务器的网络连接以及config.ini中的 `base_url` 是否正确。"})
        raise HTTPException(503, {"error": "多个上游API请求失败，原因各不相同。", "report": f"失败统计: {errors['auth']}个认证失败, {errors['rate_limit']}个速率限制, {errors['timeout']}个连接超时, {errors['other']}个其他错误。"})

    async def stream_response_content(result: dict):
        content = result["choices"][0]["message"]["content"]; resp_id, created, model = result.get("id", f"c-{int(time.time())}"), int(time.time()), result.get("model", "gemini-pro")
        async def gen():
            for i in range(0, len(content), max(1, len(content) // 50)): yield f"data: {json.dumps({'id': resp_id, 'created': created, 'model': model, 'choices': [{'delta': {'content': content[i:i + max(1, len(content) // 50)]}}]})}\n\n"; await asyncio.sleep(0.01)
            yield f"data: {json.dumps({'id': resp_id, 'created': created, 'model': model, 'choices': [{'delta': {}, 'finish_reason': 'stop'}]})}\n\n"; yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    @app_fastapi.post("/v1/chat/completions")
    async def chat_completions_proxy(req: ChatRequest, http_req: Request):
        try:
            auth = http_req.headers.get("Authorization", "").split(" ")
            if len(auth)!=2 or auth[0]!="Bearer" or auth[1]!=config_manager.get_server_config()['api_key']: raise HTTPException(401, "API密钥无效或格式不正确。")
            data = await http_req.json(); best_resp = await process_api_requests(data)
            return await stream_response_content(best_resp) if req.stream else JSONResponse(best_resp)
        except HTTPException: raise
        except Exception as e: log_and_print_error(e, "处理聊天请求时发生未知内部错误。"); raise HTTPException(500, "服务器内部错误")

    # ==================== Web UI 和配置 API ====================
    static_path = get_resource_path("static")
    templates_path = get_resource_path("templates")
    os.makedirs(static_path, exist_ok=True); os.makedirs(templates_path, exist_ok=True)
    app_fastapi.mount("/static", StaticFiles(directory=str(static_path)), name="static")
    templates = Jinja2Templates(directory=str(templates_path))
    is_api_server_running = True

    @app_fastapi.get("/", response_class=HTMLResponse)
    async def read_root_html(request: Request):
        index_path = templates_path / "index.html"
        if not index_path.exists():
            return HTMLResponse("<h1>错误：未找到界面文件</h1><p>请确保 'templates/index.html' 文件与您的Python脚本在同一目录下。</p>", status_code=404)
        return templates.TemplateResponse("index.html", {"request": request})

    @app_fastapi.get("/api/config")
    async def get_config():
        try: return {'server': config_manager.get_server_config(), 'api_keys': config_manager.get_api_keys(), 'base_url': config_manager.get_base_url()}
        except Exception as e: log_and_print_error(e, "获取配置失败。"); raise HTTPException(500, str(e))

    @app_fastapi.post("/api/config")
    async def save_config_api(request: Request):
        try:
            data = await request.json()
            if 'server' in data: config_manager.set_server_config(**data['server'])
            if 'api_keys' in data: config_manager.set_api_keys(**data['api_keys'])
            if 'base_url' in data: config_manager.set_base_url(data['base_url'])
            return {'success': True, 'message': '配置已保存并立即生效'}
        except Exception as e: log_and_print_error(e, "保存配置失败。"); raise HTTPException(500, str(e))
    
    @app_fastapi.get("/api/server/status")
    async def get_server_status(): return {'is_running': is_api_server_running}

# ==================== 主程序入口 ====================
def main():
    print("正在启动LLM代理服务 (全中文最终修复版)...")
    setup_termux_environment()
    try: import uvicorn
    except ImportError: print("[处理办法]：请运行 'pip install uvicorn' 来安装Web服务器。"); sys.exit(1)

    try:
        server_config = config_manager.get_server_config()
        web_host, web_port = server_config['web_host'], server_config['web_port']
        
        if not check_port_available(web_host, web_port):
            print(f"\n[警告]：端口 {web_port} 已被占用。")
            if is_termux_environment():
                alt_port = web_port + 1
                if check_port_available(web_host, alt_port): print(f"[处理办法]：将自动尝试使用备用端口 {alt_port}。"); server_config.update({'web_port': alt_port, 'port': alt_port}); config_manager.set_server_config(**server_config); web_port = alt_port
                else: print(f"[处理办法]：备用端口 {alt_port} 也被占用。请在 config.ini 文件中手动指定一个未被占用的端口。"); sys.exit(1)
            else: print(f"[处理办法]：请关闭占用端口 {web_port} 的程序，或在 config.ini 中修改 'web_port' 和 'port' 的值。"); sys.exit(1)

        script_name, app_string = Path(__file__).stem, f"{Path(__file__).stem}:app_fastapi"
        print(f"\n服务启动成功！")
        print(f"管理界面: http://{web_host}:{web_port}/")
        print(f"API 端点: http://{server_config['host']}:{server_config['port']}/v1/chat/completions")
        
        uvicorn.run(app_string, host=web_host, port=web_port, log_level="info", reload=False, workers=1 if is_termux_environment() else None)
    except Exception as e:
        log_and_print_error(e, "启动服务时发生严重错误，程序已退出。请根据上面的技术细节排查问题。")
        sys.exit(1)

if __name__ == "__main__":
    main()
