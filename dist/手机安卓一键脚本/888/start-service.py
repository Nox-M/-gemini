#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM代理服务启动器
简化服务启动流程
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def start_service():
    """启动服务"""
    print("🚀 启动LLM代理服务...")
    
    # 确保在正确目录
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # 检查依赖
    try:
        import fastapi
        import uvicorn
        import httpx
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: pip install fastapi uvicorn httpx pydantic python-multipart aiofiles")
        return
    
    # 启动服务
    try:
        print("📡 正在启动服务...")
        print("🌐 管理界面: http://localhost:8081")
        print("🔌 API端点: http://localhost:8080")
        print("⏳ 等待3秒后自动打开浏览器...")
        
        # 延迟打开浏览器
        def open_browser():
            time.sleep(3)
            webbrowser.open("http://localhost:8081")
        
        import threading
        threading.Thread(target=open_browser, daemon=True).start()
        
        # 启动服务
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app:app_fastapi", 
            "--host", "0.0.0.0", 
            "--port", "8081",
            "--reload"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 服务已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")

if __name__ == "__main__":
    start_service()