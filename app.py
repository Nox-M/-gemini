#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM代理服务 - Termux版
专为Android Termux环境优化的FastAPI版本
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
import traceback  # 新增：导入traceback模块，用于打印详细的错误堆栈信息
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
        base_path = Path(sys.executable).parent
    else:
        base_path = Path(__file__).parent
    return str(base_path / relative_path)

def check_port_available(host: str, port: int) -> bool:
    """检查端口是否可用"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex((host, port))
            return result != 0
    except Exception as e:
        # 增强报错：输出详细的错误信息
        logger.error(f"检查端口可用性时出错 ({host}:{port}): {e}\n{traceback.format_exc()}")
        return False

def setup_termux_environment():
    """设置Termux环境优化"""
    if not is_termux_environment():
        return
    
    try:
        subprocess.run(['termux-wake-lock'], check=False, capture_output=True)
        logger.info("已启用termux-wake-lock")
    except FileNotFoundError:
        logger.warning("termux-wake-lock未找到，服务可能会在后台被系统终止")
    
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
        self.config.clear()
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config.read_file(f)
        else:
            self.create_default_config()
    
    def create_default_config(self):
        """创建默认配置"""
        if is_termux_environment():
            default_port, default_host, default_timeout, default_min_length, default_web_port = '5000', '0.0.0.0', '120', '300', '5000'
        else:
            default_port, default_host, default_timeout, default_min_length, default_web_port = '5000', '127.0.0.1', '180', '400', '5000'
        
        self.config['SERVER'] = {'port': default_port, 'host': default_host, 'api_key': '123', 'min_response_length': default_min_length, 'request_timeout': default_timeout, 'web_port': default_web_port, 'web_host': default_host}
        self.config['API_KEYS'] = {'group1': json.dumps(["AIzaSyCgh-9h5PhprwiGSrk7oNxD5Bl240gI6Fk", "AIzaSyBmfY6uDjeDmaCbjjuDpMhLJe6H8nMMGXA", "AIzaSyCRxaB09p2wEDJPbwc69tEukfrsv0HT5YQ", "AIzaSyDJqNc2s-L2_RW0-AwMevHRvhYgEMMXLRM"]), 'group2': json.dumps(["AIzaSyDxG_Dn27XZ-OSeg_iWbGduohqD9gYrGiI", "AIzaSyDP-WGwWX4SY2uLTaKAivWwuXzX0LqSui0", "AIzaSyBwlIzbZ7bnRtYU7iicNdMnLYKkd8XVPDU", "AIzaSyDIwwW4ApVM7Dsj7BuCq4766eCWcOW9_mM"])}
        self.config['API'] = {'base_url': 'https://generativelanguage.googleapis.com/v1beta'}
        self.save_config()
    
    def save_config(self):
        """保存配置到文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                self.config.write(f)
            logger.info("配置已保存")
            self.load_config()
        except Exception as e:
            # 增强报错：输出详细的错误信息
            logger.error(f"保存配置失败: {e}\n{traceback.format_exc()}")
            raise
    
    def force_reload_config(self):
        logger.info("强制重新加载配置文件")
        self.load_config()
    
    def get_server_config(self) -> Dict[str, Any]:
        return {'port': int(self.config['SERVER']['port']), 'host': self.config['SERVER']['host'], 'api_key': self.config['SERVER']['api_key'], 'min_response_length': int(self.config['SERVER']['min_response_length']), 'request_timeout': int(self.config['SERVER']['request_timeout']), 'web_port': int(self.config['SERVER']['web_port']), 'web_host': self.config['SERVER']['web_host']}
    
    def set_server_config(self, port: int, host: str, api_key: str, min_response_length: int, request_timeout: int, web_port: int, web_host: str):
        self.config['SERVER'].update({'port': str(port), 'host': host, 'api_key': api_key, 'min_response_length': str(min_response_length), 'request_timeout': str(request_timeout), 'web_port': str(web_port), 'web_host': web_host})
        self.save_config()
    
    def get_api_keys(self) -> Dict[str, List[str]]:
        return {'group1': json.loads(self.config['API_KEYS']['group1']), 'group2': json.loads(self.config['API_KEYS']['group2'])}
    
    def set_api_keys(self, group1: List[str], group2: List[str]):
        self.config['API_KEYS'].update({'group1': json.dumps(group1), 'group2': json.dumps(group2)})
        self.save_config()
    
    def get_base_url(self) -> str:
        return self.config['API']['base_url']
    
    def set_base_url(self, base_url: str):
        self.config['API']['base_url'] = base_url
        self.save_config()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    # 增强格式：加入函数名和行号，精确定位日志来源
    format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('llm_proxy.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

config_manager = ConfigManager()

# ==================== FastAPI服务 (如果可用) ====================
if FASTAPI_AVAILABLE:
    class ChatRequest(BaseModel):
        model: str
        messages: List[Dict[str, Any]]
        temperature: float = 0.7
        max_tokens: int = 4096
        stream: bool = False

    current_group_index = 0

    def get_current_api_keys():
        global current_group_index
        config_manager.force_reload_config()
        api_keys = config_manager.get_api_keys()
        keys = api_keys['group1'] if current_group_index == 0 else api_keys['group2']
        current_group_index = 1 - current_group_index
        valid_keys = [key for key in keys if key and not key.startswith("YOUR_") and len(key) > 10]
        logger.info(f"当前使用密钥组: {'group1' if current_group_index == 1 else 'group2'}, 有效密钥数量: {len(valid_keys)}")
        return valid_keys

    async def send_single_request(client: httpx.AsyncClient, api_key: str, request_data: dict):
        cleaned_data = {k: v for k, v in request_data.items() if k in {'model', 'messages', 'temperature', 'max_tokens', 'top_p', 'top_k', 'stop'}}
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        url = f"{config_manager.get_base_url()}/openai/chat/completions"
        try:
            response = await client.post(url, headers=headers, json=cleaned_data, timeout=config_manager.get_server_config()['request_timeout'])
            response.raise_for_status()
            response_text = response.text
            if "data:" in response_text:
                lines, content, final_id, final_model, final_created = response_text.strip().split('\n'), "", "", "", int(time.time())
                for line in lines:
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            if data == "[DONE]": continue
                            if "choices" in data and data["choices"]:
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta: content += delta["content"]
                                if "id" in data: final_id = data["id"]
                                if "model" in data: final_model = data["model"]
                                if "created" in data: final_created = data["created"]
                        except json.JSONDecodeError: continue
                if content: return {"id": final_id or "chatcmpl-" + str(int(time.time())), "object": "chat.completion", "created": final_created, "model": final_model or "gemini-2.5-flash", "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
            try:
                return response.json()
            except ValueError as e:
                # 增强报错：输出详细的JSON解析错误
                logger.error(f"JSON解析错误: {e}\n响应原文: {response.text}\n{traceback.format_exc()}")
                return None
        except httpx.RequestError as e:
            # 增强报错：输出详细的网络请求错误
            logger.error(f"请求错误 (URL: {e.request.url}): {e}\n{traceback.format_exc()}")
            return None
        except Exception as e:
            # 增强报错：输出其他所有未知错误
            logger.error(f"发送单个请求时发生未知错误: {e}\n{traceback.format_exc()}")
            return None

    async def process_responses(tasks, min_length):
        valid_responses = []
        for future in asyncio.as_completed(tasks):
            try:
                result = await future
                if result and "choices" in result and result["choices"]:
                    content = result["choices"][0].get("message", {}).get("content", "")
                    if len(content) >= min_length:
                        valid_responses.append({'result': result, 'content': content, 'token_count': len(content)})
            except asyncio.CancelledError:
                pass
            except Exception as e:
                # 增强报错：即使只是一个并发任务出错，也要打印出来
                logger.error(f"处理并发任务中的一个响应时出错: {e}\n{traceback.format_exc()}")
        return valid_responses

    async def generate_fake_stream_response(request_data: dict):
        try:
            current_keys = get_current_api_keys()
            if not current_keys: raise HTTPException(status_code=500, detail="没有可用的API密钥")
            
            async with httpx.AsyncClient() as client:
                tasks = [asyncio.create_task(send_single_request(client, key, request_data)) for key in current_keys]
                valid_responses = await process_responses(tasks, config_manager.get_server_config()['min_response_length'])
                
                if valid_responses:
                    remaining_tasks = [t for t in tasks if not t.done()]
                    if remaining_tasks:
                        done, pending = await asyncio.wait(remaining_tasks, timeout=15)
                        for task in pending: task.cancel()
                        valid_responses.extend(await process_responses(done, config_manager.get_server_config()['min_response_length']))

                if valid_responses:
                    best_response = max(valid_responses, key=lambda x: x['token_count'])
                    return await stream_response_content(best_response['result'], best_response['content'])
            
            raise HTTPException(status_code=503, detail="所有上游API请求均失败")
        except HTTPException:
            raise
        except Exception as e:
            # 增强报错：输出生成流式响应时的详细错误
            logger.error(f"生成流式响应时出错: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

    async def stream_response_content(result: dict, content: str):
        response_id, created_time, model_name = result.get("id", f"chatcmpl-{int(time.time())}"), result.get("created", int(time.time())), result.get("model", "gemini-2.5-flash")
        async def generate_stream():
            chunk_size = max(1, len(content) // 50)
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i+chunk_size]
                chunk_data = {"id": response_id, "object": "chat.completion.chunk", "created": created_time, "model": model_name, "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}]}
                yield f"data: {json.dumps(chunk_data)}\n\n"
                await asyncio.sleep(0.01)
            final_data = {"id": response_id, "object": "chat.completion.chunk", "created": created_time, "model": model_name, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(final_data)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    app_fastapi = FastAPI(title="LLM代理服务", version="2.0.0")
    app_fastapi.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

    @app_fastapi.post("/v1/chat/completions")
    async def chat_completions_proxy(chat_request: ChatRequest, request: Request):
        try:
            api_key_header = request.headers.get("Authorization")
            if not api_key_header or not api_key_header.startswith("Bearer "): raise HTTPException(status_code=401, detail="缺少API密钥或格式不正确")
            provided_key = api_key_header.split(" ")[1]
            server_config = config_manager.get_server_config()
            if not provided_key or provided_key != server_config['api_key']: raise HTTPException(status_code=401, detail="API密钥无效")
            
            request_data = await request.json()
            if chat_request.stream: return await generate_fake_stream_response(request_data)
            
            current_keys = get_current_api_keys()
            if not current_keys: raise HTTPException(status_code=500, detail="服务器未配置有效的API密钥")
            
            async with httpx.AsyncClient() as client:
                tasks = [asyncio.create_task(send_single_request(client, key, request_data)) for key in current_keys]
                valid_responses = await process_responses(tasks, server_config['min_response_length'])
                
                if valid_responses:
                    remaining_tasks = [t for t in tasks if not t.done()]
                    if remaining_tasks:
                        done, pending = await asyncio.wait(remaining_tasks, timeout=15)
                        for task in pending: task.cancel()
                        valid_responses.extend(await process_responses(done, server_config['min_response_length']))

                if valid_responses:
                    best_response = max(valid_responses, key=lambda x: x['token_count'])
                    return JSONResponse(content=best_response['result'])
            
            raise HTTPException(status_code=503, detail="所有上游API请求均失败")
        except HTTPException:
            raise
        except Exception as e:
            # 增强报错：在主API入口捕获所有未预料的错误
            logger.error(f"处理聊天完成请求时出错: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

    @app_fastapi.get("/api")
    def read_root():
        return {"status": "ok", "message": "LLM代理服务正在运行"}

    static_path, templates_path = get_resource_path("static"), get_resource_path("templates")
    os.makedirs(static_path, exist_ok=True); os.makedirs(templates_path, exist_ok=True)
    app_fastapi.mount("/static", StaticFiles(directory=static_path), name="static")
    templates = Jinja2Templates(directory=templates_path)
    is_api_server_running = True

    @app_fastapi.get("/", response_class=HTMLResponse)
    async def read_root_html(request: Request): return templates.TemplateResponse("index.html", {"request": request})

    @app_fastapi.get("/api/config")
    async def get_config():
        try:
            return {'server': config_manager.get_server_config(), 'api_keys': config_manager.get_api_keys(), 'base_url': config_manager.get_base_url()}
        except Exception as e:
            # 增强报错：输出获取配置时的详细错误
            logger.error(f"获取配置时出错: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))

    @app_fastapi.post("/api/config")
    async def save_config_api(request: Request):
        try:
            data = await request.json()
            if 'server' in data: config_manager.set_server_config(**{k: int(v) if k in ['port', 'min_response_length', 'request_timeout', 'web_port'] else v for k, v in data['server'].items()})
            if 'api_keys' in data: config_manager.set_api_keys(group1=data['api_keys']['group1'], group2=data['api_keys']['group2'])
            if 'base_url' in data: config_manager.set_base_url(data['base_url'])
            return {'success': True, 'message': '配置已保存并立即生效', 'timestamp': int(time.time())}
        except Exception as e:
            # 增强报错：输出保存配置时的详细错误
            logger.error(f"保存配置时出错: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))

    @app_fastapi.post("/api/server/start")
    async def start_api_server():
        global is_api_server_running
        if is_api_server_running: raise HTTPException(status_code=400, detail='服务器已在运行中')
        try:
            is_api_server_running = True
            server_config = config_manager.get_server_config()
            return {'success': True, 'url': f"http://{server_config['host']}:{server_config['port']}"}
        except Exception as e:
            logger.error(f"启动API服务器失败: {e}\n{traceback.format_exc()}"); raise HTTPException(status_code=500, detail=str(e))

    @app_fastapi.post("/api/server/stop")
    async def stop_api_server():
        global is_api_server_running
        if not is_api_server_running: raise HTTPException(status_code=400, detail='服务器未运行')
        try:
            is_api_server_running = False
            return {'success': True}
        except Exception as e:
            logger.error(f"停止API服务器失败: {e}\n{traceback.format_exc()}"); raise HTTPException(status_code=500, detail=str(e))

    @app_fastapi.get("/api/server/status")
    async def get_server_status(): return {'is_running': is_api_server_running}

# ==================== 主程序入口 ====================
def main():
    print("正在启动LLM代理服务 - Termux版...")
    is_termux = is_termux_environment()
    if is_termux:
        print("检测到Termux环境，进行优化配置...")
        setup_termux_environment()
        server_config = config_manager.get_server_config()
        if server_config['host'] == '127.0.0.1':
            server_config['host'] = '0.0.0.0'
            config_manager.set_server_config(**server_config)
    
    server_config = config_manager.get_server_config()
    if not check_port_available(server_config['web_host'], server_config['web_port']):
        print(f"警告: 端口 {server_config['web_port']} 可能已被占用")
        if is_termux:
            alternative_port = server_config['web_port'] + 1
            if check_port_available(server_config['web_host'], alternative_port):
                print(f"尝试使用备用端口: {alternative_port}")
                server_config.update({'web_port': alternative_port, 'port': alternative_port})
                config_manager.set_server_config(**server_config)
    
    service_url = f"http://{server_config['web_host']}:{server_config['web_port']}"
    api_service_url = f"http://{server_config['host']}:{server_config['port']}"
    print(f"服务将运行在: {service_url}\nAPI端点: {api_service_url}/v1/chat/completions\n管理界面: {service_url}/")
    if is_termux:
        print("Termux环境提示:\n1. 请确保已允许Termux访问网络权限\n2. 如需从外部访问，请确保端口转发已正确配置\n3. 建议使用 'termux-wake-lock' 保持服务运行\n4. 建议使用 'termux-battery-status' 监控电池状态\n5. 可以使用 'nohup python app.py &' 在后台运行服务")
    
    try:
        import uvicorn
        # 在 __main__ block 中，app 实例需要通过字符串形式传递给 uvicorn
        # 获取当前脚本的文件名
        script_name = Path(__file__).stem
        app_string = f"{script_name}:app_fastapi"

        uvicorn_config = {"app": app_string, "host": server_config['web_host'], "port": server_config['web_port'], "log_level": "info", "reload": False}
        if is_termux: uvicorn_config.update({"workers": 1, "limit_concurrency": 10, "timeout_keep_alive": 5})
        uvicorn.run(**uvicorn_config)
    except ImportError:
        print("错误：缺少uvicorn依赖。请运行 'pip install uvicorn'")
    except Exception as e:
        # 增强报错：如果启动服务本身就失败了，打印最详细的错误信息
        print(f"启动服务失败: {e}\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()
