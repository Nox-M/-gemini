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
        model_name = result.get("model", "gemini-2.5-flash")
        
        async def generate_stream():
            chunk_size = max(1, len(content) // 50)
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i+chunk_size]
                chunk_data = {
                    "id": response_id, "object": "chat.completion.chunk", "created": created_time, "model": model_name,
                    "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                await asyncio.sleep(0.01)
            
            final_data = {
                "id": response_id, "object": "chat.completion.chunk", "created": created_time, "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(final_data)}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    # 初始化FastAPI应用
    app_fastapi = FastAPI(title="LLM代理服务", version="2.0.0")
    app_fastapi.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
    )

    # ▼▼▼▼▼ 第二个修改点 ▼▼▼▼▼
    @app_fastapi.post("/v1/chat/completions")
    async def chat_completions_proxy(chat_request: ChatRequest, request: Request):
        """
        【修改版】处理聊天请求，提供详细的错误报告。
        """
        try:
            # 检查传入的代理API密钥
            api_key_header = request.headers.get("Authorization")
            if not api_key_header or not api_key_header.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="缺少代理API密钥或格式不正确 (Missing proxy API key)")
            
            provided_key = api_key_header.split(" ")[1]
            server_config = config_manager.get_server_config()
            if not provided_key or provided_key != server_config['api_key']:
                raise HTTPException(status_code=401, detail="代理API密钥无效 (Invalid proxy API key)")
            
            request_data = await request.json()
            current_keys = get_current_api_keys()
            if not current_keys:
                raise HTTPException(status_code=500, detail="服务器未配置有效的上游API密钥 (No upstream API keys configured)")

            # 用于存储所有捕获到的错误和有效的响应
            captured_errors = []
            valid_responses = []
            
            async with httpx.AsyncClient() as client:
                tasks = [
                    asyncio.create_task(send_single_request(client, key, request_data))
                    for key in current_keys
                ]
                
                # 并发处理所有请求
                for future in asyncio.as_completed(tasks):
                    try:
                        # 直接获取原始的 httpx.Response 对象
                        response = await future
                        
                        # 成功获取响应后，解析内容
                        result = response.json()
                        
                        if result and "choices" in result and result["choices"]:
                            message_content = result["choices"][0].get("message", {}).get("content", "")
                            if len(message_content) >= server_config['min_response_length']:
                                valid_responses.append({
                                    'result': result,
                                    'content': message_content,
                                    'token_count': len(message_content)
                                })
                    except httpx.HTTPStatusError as e:
                        # 捕获到来自Google的HTTP错误
                        logger.error(f"上游API错误: 状态码 {e.response.status_code}, 响应: {e.response.text}")
                        captured_errors.append(e)
                    except httpx.RequestError as e:
                        # 捕获到网络层面的错误
                        logger.error(f"网络请求错误: {e}")
                        captured_errors.append(e)
                    except Exception as e:
                        # 捕获其他未知错误
                        logger.error(f"处理响应时发生未知错误: {e}")
                        captured_errors.append(e)

            # 如果有任何一个成功的响应，就按原逻辑走
            if valid_responses:
                best_response = max(valid_responses, key=lambda x: x['token_count'])
                if chat_request.stream:
                    return await stream_response_content(best_response['result'], best_response['content'])
                else:
                    return JSONResponse(content=best_response['result'])

            # 如果没有成功响应，则分析错误并返回最详细的那个
            if not captured_errors:
                # 这种情况很少见，比如没有有效的密钥
                raise HTTPException(status_code=503, detail="所有上游API请求均失败，且未收到任何错误响应。")

            # 优先返回最常见的错误类型
            best_error = None
            for error in captured_errors:
                if isinstance(error, httpx.HTTPStatusError) and error.response.status_code == 429: # Too Many Requests
                    best_error = error
                    break
            if not best_error:
                for error in captured_errors:
                    if isinstance(error, httpx.HTTPStatusError) and error.response.status_code == 400: # Bad Request (often invalid key)
                        best_error = error
                        break
            if not best_error:
                best_error = captured_errors[0] # 随便选一个

            # 将捕获到的最详细错误返回给SillyTavern
            if isinstance(best_error, httpx.HTTPStatusError):
                # 将Google返回的原始错误信息直接透传回去
                try:
                    error_json = best_error.response.json()
                    return JSONResponse(status_code=best_error.response.status_code, content=error_json)
                except json.JSONDecodeError:
                    return JSONResponse(status_code=best_error.response.status_code, content={"error": {"message": best_error.response.text}})
            else:
                # 对于网络等其他错误，返回一个504网关超时
                raise HTTPException(status_code=504, detail=f"代理网关错误 (Proxy Gateway Error): {str(best_error)}")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"处理聊天完成请求时出错: {e}")
            raise HTTPException(status_code=500, detail=f"服务器内部错误 (Internal Server Error): {str(e)}")
    # ▲▲▲▲▲ 第二个修改点结束 ▲▲▲▲▲

    @app_fastapi.get("/api")
    def read_root():
        return {"status": "ok", "message": "LLM代理服务正在运行"}

# ==================== FastAPI Web界面 ====================
# 设置静态文件和模板
static_path = get_resource_path("static")
templates_path = get_resource_path("templates")

# 确保路径存在
os.makedirs(static_path, exist_ok=True)
os.makedirs(templates_path, exist_ok=True)

app_fastapi.mount("/static", StaticFiles(directory=static_path), name="static")
templates = Jinja2Templates(directory=templates_path)

# 全局变量
is_api_server_running = True

@app_fastapi.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """主页"""
    return templates.TemplateResponse("index.html", {"request": request})

@app_fastapi.get("/api/config")
async def get_config():
    """获取配置"""
    try:
        server_config = config_manager.get_server_config()
        api_keys = config_manager.get_api_keys()
        base_url = config_manager.get_base_url()
        
        return {
            'server': server_config,
            'api_keys': api_keys,
            'base_url': base_url
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app_fastapi.post("/api/config")
async def save_config(request: Request):
    """保存配置"""
    try:
        data = await request.json()
        
        # 保存服务器配置
        if 'server' in data:
            server = data['server']
            config_manager.set_server_config(
                port=int(server['port']),
                host=server['host'],
                api_key=server['api_key'],
                min_response_length=int(server['min_response_length']),
                request_timeout=int(server['request_timeout']),
                web_port=int(server['web_port']),
                web_host=server['web_host']
            )
        
        # 保存API密钥
        if 'api_keys' in data:
            api_keys = data['api_keys']
            config_manager.set_api_keys(
                group1=api_keys['group1'],
                group2=api_keys['group2']
            )
        
        # 保存基础URL
        if 'base_url' in data:
            config_manager.set_base_url(data['base_url'])
        
        # 返回成功消息，包含配置更新时间戳
        return {
            'success': True,
            'message': '配置已保存并立即生效',
            'timestamp': int(time.time())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app_fastapi.post("/api/server/start")
async def start_api_server():
    """启动API服务器"""
    global is_api_server_running
    
    if is_api_server_running:
        raise HTTPException(status_code=400, detail='服务器已在运行中')
    
    try:
        is_api_server_running = True
        server_config = config_manager.get_server_config()
        
        return {
            'success': True,
            'url': f"http://{server_config['host']}:{server_config['port']}"
        }
    except Exception as e:
        logger.error(f"启动API服务器失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app_fastapi.post("/api/server/stop")
async def stop_api_server():
    """停止API服务器"""
    global is_api_server_running
    
    if not is_api_server_running:
        raise HTTPException(status_code=400, detail='服务器未运行')
    
    try:
        is_api_server_running = False
      return {'success': True}
    except Exception as e:
        logger.error(f"停止API服务器失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app_fastapi.get("/api/server/status")
async def get_server_status():
    """获取服务器状态"""
    return {
        'is_running': is_api_server_running
    }

# ==================== 主程序入口 ====================
def main():
    """主程序入口"""
    print("正在启动LLM代理服务 - Termux版...")
    
    # 检测并设置Termux环境
    is_termux = is_termux_environment()
    
    if is_termux:
        print("检测到Termux环境，进行优化配置...")
        setup_termux_environment()
        
        # 在Termux环境中，使用更保守的配置
        server_config = config_manager.get_server_config()
        if server_config['host'] == '127.0.0.1':
            server_config['host'] = '0.0.0.0'
            config_manager.set_server_config(
                port=server_config['port'],
                host=server_config['host'],
                api_key=server_config['api_key'],
                min_response_length=server_config['min_response_length'],
                request_timeout=server_config['request_timeout'],
                web_port=server_config['web_port'],
                web_host=server_config['web_host']
            )
    
    server_config = config_manager.get_server_config()
    
    # 检查端口是否可用
    if not check_port_available(server_config['web_host'], server_config['web_port']):
        print(f"警告: 端口 {server_config['web_port']} 可能已被占用")
        # 在Termux环境中尝试使用备用端口
        if is_termux:
            alternative_port = server_config['web_port'] + 1
            if check_port_available(server_config['web_host'], alternative_port):
                print(f"尝试使用备用端口: {alternative_port}")
                server_config['web_port'] = alternative_port
                server_config['port'] = alternative_port
                config_manager.set_server_config(
                    port=server_config['port'],
                    host=server_config['host'],
                    api_key=server_config['api_key'],
                    min_response_length=server_config['min_response_length'],
                    request_timeout=server_config['request_timeout'],
                    web_port=server_config['web_port'],
                    web_host=server_config['web_host']
                )
    
    # 构建服务URL
    service_url = f"http://{server_config['web_host']}:{server_config['web_port']}"
    api_service_url = f"http://{server_config['host']}:{server_config['port']}"
    
    print(f"服务将运行在: {service_url}")
    print(f"API端点: {api_service_url}/v1/chat/completions")
    print(f"管理界面: {service_url}/")
    
    if is_termux:
        print("Termux环境提示:")
        print("1. 请确保已允许Termux访问网络权限")
        print("2. 如需从外部访问，请确保端口转发已正确配置")
        print("3. 建议使用 'termux-wake-lock' 保持服务运行")
        print("4. 建议使用 'termux-battery-status' 监控电池状态")
        print("5. 可以使用 'nohup python app.py &' 在后台运行服务")
    
    # 导入uvicorn并启动服务
    try:
        import uvicorn
        
        # Termux环境特定的uvicorn配置
        uvicorn_config = {
            "app": "app_fastapi:app_fastapi", # 修改为字符串形式以支持热重载
            "host": server_config['web_host'],
            "port": server_config['web_port'],
            "log_level": "info",
            "reload": False,  # 在Termux环境中禁用热重载以提高稳定性
        }
        
        if is_termux:
            # Termux环境下使用更保守的配置
            uvicorn_config.update({
                "workers": 1,  # 单工作进程
                "limit_concurrency": 10,  # 限制并发连接数
                "timeout_keep_alive": 5,  # 较短的keep-alive超时
            })
        
        # 注意：直接运行 uvicorn.run() 时，app 参数应为 FastAPI 实例，而不是字符串
        # 如果要使用字符串以支持热重载，则应从命令行运行 uvicorn
        uvicorn.run(app_fastapi, **{k: v for k, v in uvicorn_config.items() if k != 'app'})
        
    except ImportError:
        print("错误：缺少uvicorn依赖。请运行 'pip install uvicorn'")
        return
    except Exception as e:
        print(f"启动服务失败: {e}")
        return

if __name__ == "__main__":
    main()
```
    
