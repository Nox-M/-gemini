#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM代理服务 - Web版
将GUI改为HTML界面，支持热重载更改配置，为迁移到安卓上的termux做准备
"""

import asyncio
import httpx
import os
import sys
import json
import time
import logging
import configparser
import threading
import signal
import platform
import socket
import subprocess
import webbrowser
from pathlib import Path
from typing import List, Dict, Any

# 尝试导入Flask相关模块
try:
    from flask import Flask, render_template, request, jsonify, send_from_directory
    from flask_cors import CORS
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# 尝试导入FastAPI和Pydantic（用于API代理服务）
try:
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# ==================== 辅助函数 ====================
def get_resource_path(relative_path: str) -> str:
    """获取资源的绝对路径，兼容开发环境和PyInstaller打包环境"""
    if getattr(sys, 'frozen', False):
        base_path = Path(sys.executable).parent
    else:
        base_path = Path(__file__).parent
    return str(base_path / relative_path)

def format_results_table(results: List[Dict[str, Any]], min_length: int) -> str:
    """将API请求结果格式化为包含建议的中文日志表格"""
    if not results:
        return "没有可供分析的请求结果。"

    headers = ["API密钥 (掩码)", "状态", "HTTP状态码", "内容长度", "原因/信息", "处理建议"]
    
    table_data = []
    for r in results:
        status = r.get('status', '未知')
        http_code = r.get('http_code')
        message = str(r.get('message', 'N/A'))
        content_length = r.get('content_length', 0)
        
        # --- 生成处理建议 ---
        suggestion = ""
        if status == 'success':
            if content_length >= min_length:
                suggestion = f"响应有效 (>{min_length}), 已满足要求"
            else:
                suggestion = f"内容过短 (不足{min_length}), 可能被安全策略拦截或模型输出少"
        elif status == 'error':
            if http_code == 500:
                suggestion = "上游服务器内部错误, 建议稍后重试或更换密钥"
            elif http_code in [401, 403]:
                suggestion = "API密钥无效或无权限, 请立即检查或更换此密钥"
            elif http_code == 429:
                suggestion = "请求频率过高 (限流), 建议等待或轮换未使用过的密钥"
            elif http_code is not None and 400 <= http_code < 500:
                suggestion = "客户端请求错误 (如参数错误), 请检查请求体"
            elif 'Timeout' in message:
                suggestion = "请求超时, 检查网络或在config.ini中增加'request_timeout'"
            elif 'ConnectError' in message:
                suggestion = "网络连接失败, 请检查Termux或设备的网络环境"
            elif 'JSON Decode Error' in message:
                suggestion = "API返回非JSON格式, 可能为HTML错误页, 检查上游API状态"
            else:
                suggestion = "未知错误, 请重点关注日志中关于此密钥的详细错误信息"
        
        table_data.append([
            r.get('key', 'N/A'),
            "成功" if status == 'success' else "失败",
            str(http_code) if http_code is not None else "N/A",
            str(content_length),
            message,
            suggestion
        ])

    # 计算列宽
    col_widths = [len(h) for h in headers]
    for row in table_data:
        for i, cell in enumerate(row):
            # 对中文字符宽度进行简单校正 (一个中文约等于1.8个英文字符宽)
            cell_width = sum(1.8 if '\u4e00' <= char <= '\u9fff' else 1 for char in cell)
            if cell_width > col_widths[i]:
                col_widths[i] = int(cell_width)

    # 打印表格
    separator = "+-" + "-+-".join(["-" * w for w in col_widths]) + "-+"
    header_line = "| " + " | ".join([h.center(w) for h, w in zip(headers, col_widths)]) + " |"
    
    table = [separator, header_line, separator]
    for row in table_data:
        # 填充单元格时也需要考虑中文字符
        cells_padded = []
        for i, cell in enumerate(row):
            cell_width = sum(1.8 if '\u4e00' <= char <= '\u9fff' else 1 for char in cell)
            padding = col_widths[i] - int(cell_width)
            cells_padded.append(cell + ' ' * padding)
        table.append("| " + " | ".join(cells_padded) + " |")

    table.append(separator)
    return "\n".join(table)


# ==================== 配置管理器 ====================
# (这部分代码保持不变，此处省略)
class ConfigManager:
    """配置文件管理器"""
    
    def __init__(self, config_file: str = "config.ini"):
        self.config_file = get_resource_path(config_file)
        self.config = configparser.ConfigParser()
        self.load_config()
    
    def load_config(self):
        """加载配置文件"""
        if os.path.exists(self.config_file):
            self.config.read(self.config_file, encoding='utf-8')
        else:
            self.create_default_config()
    
    def create_default_config(self):
        """创建默认配置"""
        self.config['SERVER'] = {
            'port': '8080',
            'host': '0.0.0.0',
            'api_key': '123',
            'min_response_length': '400',
            'request_timeout': '180',
            'web_port': '5001',
            'web_host': '127.0.0.1'
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
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            raise
    
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
# (send_single_request 和 其他FastAPI部分与上一版相同，此处省略以保持简洁)
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
        api_keys = config_manager.get_api_keys()
        
        if current_group_index == 0:
            keys = api_keys['group1']
            current_group_index = 1
        else:
            keys = api_keys['group2']
            current_group_index = 0
        
        valid_keys = [key for key in keys if key and not key.startswith("YOUR_") and len(key) > 10]
        return valid_keys

    async def send_single_request(client: httpx.AsyncClient, api_key: str, request_data: dict):
        """使用单个API密钥发送请求，并返回包含详细状态的字典"""
        masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else api_key
        
        cleaned_data = {}
        supported_params = {'model', 'messages', 'temperature', 'max_tokens', 'top_p', 'top_k', 'stop'}
        for key, value in request_data.items():
            if key in supported_params:
                cleaned_data[key] = value
        
        headers = { "Authorization": f"Bearer {api_key}", "Content-Type": "application/json" }
        url = f"{config_manager.get_base_url()}/openai/chat/completions"
        
        try:
            response = await client.post(url, headers=headers, json=cleaned_data, timeout=config_manager.get_server_config()['request_timeout'])
            
            response_text = response.text
            
            if response.is_error:
                logger.error(f"API Key [{masked_key}] 请求失败 - HTTP Status: {response.status_code}, Response: {response_text[:200]}")
                return { "key": masked_key, "status": "error", "http_code": response.status_code, "message": f"HTTP Error: {response.status_code}", "content_length": len(response_text), "response": None }

            # 尝试解析JSON
            try:
                json_response = response.json()
                content = ""
                # 兼容流式和非流式返回的解析逻辑
                if "choices" in json_response and json_response["choices"]:
                    message = json_response["choices"][0].get("message", {})
                    if message:
                        content = message.get("content", "")

                return { "key": masked_key, "status": "success", "http_code": response.status_code, "message": "OK", "content_length": len(content), "response": json_response }
            except json.JSONDecodeError:
                logger.error(f"API Key [{masked_key}] 成功返回但JSON解析错误, Response: {response_text[:200]}")
                return { "key": masked_key, "status": "error", "http_code": response.status_code, "message": "JSON Decode Error", "content_length": len(response_text), "response": None }

        except httpx.TimeoutException as e:
            logger.error(f"API Key [{masked_key}] 请求超时: {e}")
            return { "key": masked_key, "status": "error", "http_code": None, "message": f"Timeout Error", "content_length": 0, "response": None }
        except httpx.RequestError as e:
            logger.error(f"API Key [{masked_key}] 请求错误: {e}")
            return { "key": masked_key, "status": "error", "http_code": None, "message": f"Request Error: {type(e).__name__}", "content_length": 0, "response": None }
        except Exception as e:
            logger.error(f"API Key [{masked_key}] 未知错误: {e}")
            return { "key": masked_key, "status": "error", "http_code": None, "message": f"Unknown Error: {type(e).__name__}", "content_length": 0, "response": None }
    
    async def generate_fake_stream_response(request_data: dict):
        """获取完整的响应内容，等待15秒后选择token最长的响应，然后以流式方式发送给前端"""
        # (此部分逻辑保持不变，但为了简洁可以暂时忽略)
        pass

    # 初始化FastAPI应用
    app_fastapi = FastAPI(title="LLM代理服务", version="2.0.0")
    app_fastapi.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
    )

    @app_fastapi.post("/v1/chat/completions")
    async def chat_completions_proxy(chat_request: ChatRequest, request: Request):
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
                raise HTTPException(status_code=501, detail="流式响应的详细日志记录尚未实现")

            current_keys = get_current_api_keys()
            if not current_keys:
                raise HTTPException(status_code=500, detail="服务器未配置有效的API密钥")
            
            all_results = []
            
            async with httpx.AsyncClient() as client:
                tasks = [asyncio.create_task(send_single_request(client, key, request_data)) for key in current_keys]
                
                for future in asyncio.as_completed(tasks):
                    result = await future
                    if result:
                        all_results.append(result)

            valid_responses = [
                r for r in all_results
                if r['status'] == 'success' and r['content_length'] >= server_config['min_response_length']
            ]
            
            if valid_responses:
                best_response = max(valid_responses, key=lambda x: x['content_length'])
                return JSONResponse(content=best_response['response'])
            else:
                # *** 关键改动在这里 ***
                # 调用新的表格生成函数，并传入最小长度阈值
                table_log = format_results_table(all_results, server_config['min_response_length'])
                logger.warning(f"所有上游API请求均未返回有效响应。诊断详情如下：\n{table_log}")
                raise HTTPException(status_code=503, detail="所有上游API请求均失败")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"处理聊天完成请求时出错: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

    @app_fastapi.get("/")
    def read_root():
        return {"status": "ok", "message": "LLM代理服务正在运行"}


# ==================== Flask Web界面 (如果可用) ====================
# (这部分代码保持不变，此处省略)
if FLASK_AVAILABLE:
    app_flask = Flask(__name__, 
                     template_folder='templates',
                     static_folder='static')
    CORS(app_flask)
    socketio = SocketIO(app_flask, cors_allowed_origins="*")
    
    # 全局变量
    api_server_thread = None
    is_api_server_running = False
    api_server_lock = threading.Lock()
    
    @app_flask.route('/')
    def index():
        """主页"""
        return render_template('index.html')
    
    @app_flask.route('/test')
    def test():
        """测试页面"""
        return render_template('test.html')
    
    @app_flask.route('/api/config', methods=['GET'])
    def get_config():
        """获取配置"""
        try:
            server_config = config_manager.get_server_config()
            api_keys = config_manager.get_api_keys()
            base_url = config_manager.get_base_url()
            
            return jsonify({
                'server': server_config,
                'api_keys': api_keys,
                'base_url': base_url
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app_flask.route('/api/config', methods=['POST'])
    def save_config():
        """保存配置"""
        try:
            data = request.get_json()
            
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
            
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app_flask.route('/api/server/start', methods=['POST'])
    def start_api_server():
        """启动API服务器"""
        global api_server_thread, is_api_server_running
        
        with api_server_lock:
            if is_api_server_running:
                return jsonify({'error': '服务器已在运行中'})
            
            if not FASTAPI_AVAILABLE:
                return jsonify({'error': 'FastAPI不可用，无法启动API服务器'})
            
            try:
                server_config = config_manager.get_server_config()
                
                def run_api_server():
                    try:
                        import uvicorn
                        uvicorn.run(
                            app_fastapi,
                            host=server_config['host'],
                            port=server_config['port'],
                            log_level="info"
                        )
                    except Exception as e:
                        logger.error(f"API服务器运行错误: {e}")
                        with api_server_lock:
                            global is_api_server_running
                            is_api_server_running = False
                
                api_server_thread = threading.Thread(target=run_api_server, daemon=True)
                api_server_thread.start()
                is_api_server_running = True
                
                # 通过SocketIO通知前端
                socketio.emit('server_status', {
                    'status': 'running',
                    'url': f"http://{server_config['host']}:{server_config['port']}"
                })
                
                return jsonify({'success': True})
            except Exception as e:
                logger.error(f"启动API服务器失败: {e}")
                return jsonify({'error': str(e)}), 500
    
    @app_flask.route('/api/server/stop', methods=['POST'])
    def stop_api_server():
        """停止API服务器"""
        global is_api_server_running
        
        with api_server_lock:
            if not is_api_server_running:
                return jsonify({'error': '服务器未运行'})
            
            try:
                # 注意：这里只是简单标记为停止，实际需要更复杂的进程管理
                is_api_server_running = False
                
                # 通过SocketIO通知前端
                socketio.emit('server_status', {
                    'status': 'stopped'
                })
                
                return jsonify({'success': True})
            except Exception as e:
                logger.error(f"停止API服务器失败: {e}")
                return jsonify({'error': str(e)}), 500
    
    @app_flask.route('/api/server/status', methods=['GET'])
    def get_server_status():
        """获取服务器状态"""
        with api_server_lock:
            return jsonify({
                'is_running': is_api_server_running
            })
    
    @socketio.on('connect')
    def handle_connect():
        """客户端连接时的处理"""
        emit('server_status', {
            'status': 'running' if is_api_server_running else 'stopped'
        })

# ==================== 主程序入口 ====================
# (这部分代码保持不变，此处省略)
def main():
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == 'cli':
        if not FASTAPI_AVAILABLE:
            print("错误：缺少运行命令行服务所需的FastAPI依赖。")
            print("请运行 'pip install fastapi uvicorn httpx pydantic python-multipart'")
            return
            
        print("正在以命令行模式启动API服务...")
        server_config = config_manager.get_server_config()
        
        # 导入uvicorn
        try:
            import uvicorn
            uvicorn.run(app_fastapi, host=server_config['host'], port=server_config['port'])
        except ImportError:
            print("错误：缺少uvicorn依赖。请运行 'pip install uvicorn'")
            return
    else:
        if not FLASK_AVAILABLE:
            print("错误：缺少运行Web界面所需的Flask依赖。")
            print("请运行 'pip install flask flask-cors flask-socketio'")
            if FASTAPI_AVAILABLE:
                print("你可以使用 'python app.py cli' 来运行命令行版本。")
            return
        
        print("正在启动Web界面...")
        server_config = config_manager.get_server_config()
        
        # 构建Web界面URL
        web_url = f"http://{server_config['web_host']}:{server_config['web_port']}"
        
        # 启动Flask应用前，延迟1秒后自动打开浏览器
        def open_browser():
            time.sleep(1.5)  # 等待服务器启动
            try:
                webbrowser.open(web_url)
                print(f"已自动打开浏览器: {web_url}")
            except Exception as e:
                print(f"自动打开浏览器失败: {e}")
                print(f"请手动访问: {web_url}")
        
        # 在新线程中打开浏览器，避免阻塞主线程
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()
        
        # 启动Flask应用
        print(f"Web界面将运行在: {web_url}")
        socketio.run(
            app_flask,
            host=server_config['web_host'],
            port=server_config['web_port'],
            debug=True,  # 启用调试模式，支持热重载
            allow_unsafe_werkzeug=True # 添加此参数以解决报错
        )

if __name__ == "__main__":
    main()
