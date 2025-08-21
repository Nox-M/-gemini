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
    except Exception:
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
            default_port = '5000'
            default_host = '0.0.0.0'
            default_timeout = '120'
            default_min_length = '300'
            default_web_port = '5000'
        else:
            default_port = '5000'
            default_host = '127.0.0.1'
            default_timeout = '180'
            default_min_length = '400'
            default_web_port = '5000'
        
        self.config['SERVER'] = {
            'port': default_port, 'host': default_host, 'api_key': '123',
            'min_response_length': default_min_length, 'request_timeout': default_timeout,
            'web_port': default_web_port, 'web_host': default_host
        }
        
        self.config['API_KEYS'] = {
            'group1': json.dumps(["AIzaSyC..."]),
            'group2': json.dumps(["AIzaSyD..."])
        }
        
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
            logger.error(f"保存配置失败: {e}")
            raise
    
    def force_reload_config(self):
        """强制重新加载配置文件"""
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
    format='%(asctime)s - %(levelname)s - %(message)s',
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

    current_group_index = 0

    def get_current_api_keys():
        """根据轮询机制返回当前应该使用的API密钥组"""
        global current_group_index
        config_manager.force_reload_config()
        api_keys = config_manager.get_api_keys()
        
        if current_group_index == 0:
            keys = api_keys['group1']
            current_group_index = 1
        else:
            keys = api_keys['group2']
            current_group_index = 0
        
        valid_keys = [key for key in keys if key and not key.startswith("YOUR_") and len(key) > 10]
        logger.info(f"当前使用密钥组: {'group1' if current_group_index == 1 else 'group2'}, 有效密钥数量: {len(valid_keys)}")
        return valid_keys
    
    # NEW/MODIFIED: Enhanced request function to return detailed status
    async def send_single_request(client: httpx.AsyncClient, api_key: str, request_data: dict):
        """使用单个API密钥发送请求，并返回详细的结果字典"""
        masked_key = f"{api_key[:6]}...{api_key[-4:]}"
        
        cleaned_data = {}
        supported_params = {'model', 'messages', 'temperature', 'max_tokens', 'top_p', 'top_k', 'stop'}
        for key, value in request_data.items():
            if key in supported_params:
                cleaned_data[key] = value
        
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        url = f"{config_manager.get_base_url()}/openai/chat/completions"
        
        try:
            response = await client.post(url, headers=headers, json=cleaned_data, timeout=config_manager.get_server_config()['request_timeout'])
            response.raise_for_status()
            
            # This part handles non-standard stream responses sent as a single block
            response_text = response.text
            if "data:" in response_text:
                content = ""
                for line in response_text.strip().split('\n'):
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            if data == "[DONE]": continue
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            if "content" in delta: content += delta["content"]
                        except (json.JSONDecodeError, IndexError): continue
                
                if content:
                    full_response = {
                        "id": "chatcmpl-" + str(int(time.time())), "object": "chat.completion",
                        "choices": [{"message": {"content": content}}]
                    }
                    return {"status": "success", "http_code": response.status_code, "data": full_response, "key": masked_key}
            
            return {"status": "success", "http_code": response.status_code, "data": response.json(), "key": masked_key}

        except httpx.TimeoutException:
            return {"status": "error", "http_code": 408, "message": "请求超时", "key": masked_key}
        except httpx.HTTPStatusError as e:
            error_message = f"HTTP {e.response.status_code}"
            try:
                error_detail = e.response.json().get("error", {}).get("message", "No details")
                error_message += f": {error_detail[:80]}" # Limit error length
            except (json.JSONDecodeError, AttributeError):
                pass
            return {"status": "error", "http_code": e.response.status_code, "message": error_message, "key": masked_key}
        except httpx.RequestError as e:
            return {"status": "error", "http_code": None, "message": f"网络连接错误", "key": masked_key}
        except json.JSONDecodeError:
             return {"status": "error", "http_code": response.status_code, "message": "响应JSON解析失败", "key": masked_key}
        except Exception as e:
            return {"status": "error", "http_code": None, "message": f"未知错误: {str(e)[:50]}", "key": masked_key}

    # NEW/MODIFIED: Function to log the summary table
    def log_request_summary(results: List[Dict]):
        """在日志中以表格形式输出所有API请求的摘要"""
        if not results: return

        logger.info("========== API 请求结果摘要 ==========")
        header = f"| {'API密钥':<15} | {'状态码':<6} | {'结果':<18} | {'内容长度':<10} | {'处理建议':<55} |"
        separator = f"|{'='*17}|{'='*8}|{'='*20}|{'='*12}|{'='*57}|"
        
        logger.info(header)
        logger.info(separator)

        for res in results:
            key = res.get('key', 'N/A')
            code = str(res.get('http_code', 'N/A'))
            result = res.get('result_text', '未知')
            length = str(res.get('content_length', 'N/A'))
            suggestion = res.get('suggestion', '')
            logger.info(f"| {key:<15} | {code:<6} | {result:<18} | {length:<10} | {suggestion:<55} |")
        
        logger.info(separator)
    
    # NEW/MODIFIED: Refactored main logic to process requests and generate summaries
    async def process_and_summarize_requests(request_data: dict) -> List[Dict]:
        """Processes a request against all API keys, logs a summary, and returns all results."""
        current_keys = get_current_api_keys()
        if not current_keys:
            logger.warning("服务器未配置任何有效的API密钥")
            return []
        
        all_results = []
        min_len = config_manager.get_server_config()['min_response_length']

        async with httpx.AsyncClient() as client:
            tasks = [asyncio.create_task(send_single_request(client, key, request_data)) for key in current_keys]
            raw_results = await asyncio.gather(*tasks)

            for res in raw_results:
                if res['status'] == 'success':
                    content = res['data'].get("choices", [{}])[0].get("message", {}).get("content", "")
                    content_length = len(content)
                    res['content_length'] = content_length
                    
                    if content_length >= min_len:
                        res['result_text'] = "成功"
                        res['suggestion'] = "响应有效，可被选用。"
                    else:
                        res['result_text'] = f"内容过短 ({content_length}) B"
                        res['suggestion'] = f"响应长度未达到最小要求({min_len})，将被忽略。"
                else: # Handle all error cases
                    res['content_length'] = 0
                    code = res['http_code']
                    if code in [401, 403]:
                        res['result_text'] = "认证失败"
                        res['suggestion'] = "API密钥无效或已过期，请检查或更换。"
                    elif code == 429:
                        res['result_text'] = "速率超限"
                        res['suggestion'] = "密钥请求过于频繁，请等待或更换。"
                    elif code == 408:
                        res['result_text'] = "请求超时"
                        res['suggestion'] = "网络问题或上游响应慢，可尝试增加`request_timeout`。"
                    elif code == 500:
                        res['result_text'] = "上游服务器错误"
                        res['suggestion'] = "Google API端可能存在问题，请稍后再试。"
                    else:
                        res['result_text'] = "请求失败"
                        res['suggestion'] = "检查网络、Base URL配置，以及错误信息: " + res['message']
                all_results.append(res)
        
        log_request_summary(all_results)
        return all_results

    async def generate_fake_stream_response(request_data: dict):
        """(MODIFIED) Generates stream response using the new processing logic"""
        all_results = await process_and_summarize_requests(request_data)
        
        valid_responses = [r for r in all_results if r.get('result_text') == "成功"]

        if valid_responses:
            best_response = max(valid_responses, key=lambda x: x['content_length'])
            content = best_response['data']['choices'][0]['message']['content']
            return await stream_response_content(best_response['data'], content)

        raise HTTPException(status_code=503, detail="所有上游API请求均失败或响应不满足要求 (详见服务器日志)")


    async def stream_response_content(result: dict, content: str):
        """将完整的响应内容以流式方式发送给前端"""
        # ... (This function remains unchanged)
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

    app_fastapi = FastAPI(title="LLM代理服务", version="2.0.0")
    app_fastapi.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
    )

    @app_fastapi.post("/v1/chat/completions")
    async def chat_completions_proxy(chat_request: ChatRequest, request: Request):
        """(MODIFIED) Main proxy endpoint using the new processing logic"""
        try:
            api_key_header = request.headers.get("Authorization")
            if not api_key_header or not api_key_header.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="缺少API密钥或格式不正确")
            
            provided_key = api_key_header.split(" ")[1]
            server_config = config_manager.get_server_config()
            if not provided_key or provided_key != server_config['api_key']:
                raise HTTPException(status_code=401, detail="API密钥无效")
            
            request_data = await request.json()
            
            if chat_request.stream:
                return await generate_fake_stream_response(request_data)

            # Non-streaming logic
            all_results = await process_and_summarize_requests(request_data)
            valid_responses = [r for r in all_results if r.get('result_text') == "成功"]
            
            if valid_responses:
                best_response = max(valid_responses, key=lambda x: x['content_length'])
                return JSONResponse(content=best_response['data'])

            raise HTTPException(status_code=503, detail="所有上游API请求均失败或响应不满足要求 (详见服务器日志)")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"处理聊天完成请求时出错: {e}")
            raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

    @app_fastapi.get("/api")
    def read_root():
        return {"status": "ok", "message": "LLM代理服务正在运行"}

    # ==================== FastAPI Web界面 (Code is unchanged) ====================
    # ... (The entire web interface section remains the same)
    # Setup static files and templates
    static_path = get_resource_path("static")
    templates_path = get_resource_path("templates")

    os.makedirs(static_path, exist_ok=True)
    os.makedirs(templates_path, exist_ok=True)

    app_fastapi.mount("/static", StaticFiles(directory=static_path), name="static")
    templates = Jinja2Templates(directory=templates_path)

    is_api_server_running = True

    @app_fastapi.get("/", response_class=HTMLResponse)
    async def read_root_web(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})
    # ... (other web interface endpoints remain unchanged)
    
# ==================== 主程序入口 ====================
def main():
    """(MODIFIED)主程序入口，修正了uvicorn的调用方式"""
    print("正在启动LLM代理服务 - Termux版...")
    is_termux = is_termux_environment()
    if is_termux:
        print("检测到Termux环境，进行优化配置...")
        setup_termux_environment()
    
    server_config = config_manager.get_server_config()
    
    # ... (Port checking and URL printing logic remains unchanged)
    
    print(f"服务将运行在: http://{server_config['web_host']}:{server_config['web_port']}")
    print(f"API端点: http://{server_config['host']}:{server_config['port']}/v1/chat/completions")
    
    try:
        import uvicorn
        # For programmatic execution, pass the app object directly.
        uvicorn.run(
            app_fastapi,
            host=server_config['web_host'],
            port=server_config['web_port'],
            log_level="info",
            workers=1 if is_termux else None # In Termux, strictly use 1 worker for stability
        )
    except ImportError:
        print("错误：缺少uvicorn依赖。请运行 'pip install uvicorn'")
    except Exception as e:
        print(f"启动服务失败: {e}")

if __name__ == "__main__":
    main()
