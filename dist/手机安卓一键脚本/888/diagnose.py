#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM代理服务诊断工具
用于检查服务状态和解决常见问题
"""

import os
import sys
import socket
import requests
import subprocess
from pathlib import Path

def check_port(host, port):
    """检查端口是否可用"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex((host, port))
            return result == 0  # 返回True表示端口被占用
    except Exception:
        return False

def check_service_health(url):
    """检查服务健康状态"""
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except Exception as e:
        return False, str(e)

def diagnose():
    """运行诊断"""
    print("🔍 LLM代理服务诊断工具")
    print("=" * 50)
    
    # 检查当前目录
    current_dir = Path.cwd()
    print(f"📁 当前工作目录: {current_dir}")
    
    # 检查必要文件
    required_files = [
        "app.py",
        "config.ini", 
        "templates/index.html",
        "static/js/app.js"
    ]
    
    print("\n📋 检查必要文件:")
    for file in required_files:
        path = Path(file)
        if path.exists():
            print(f"  ✅ {file} 存在")
        else:
            print(f"  ❌ {file} 缺失")
    
    # 检查端口
    print("\n🔌 检查端口状态:")
    ports_to_check = [8080, 8081]
    
    for port in ports_to_check:
        if check_port('localhost', port):
            print(f"  ⚠️  端口 {port} 已被占用")
        else:
            print(f"  ✅ 端口 {port} 可用")
    
    # 检查Python依赖
    print("\n📦 检查Python依赖:")
    try:
        import fastapi
        print("  ✅ FastAPI 已安装")
    except ImportError:
        print("  ❌ FastAPI 未安装")
    
    try:
        import uvicorn
        print("  ✅ Uvicorn 已安装")
    except ImportError:
        print("  ❌ Uvicorn 未安装")
    
    try:
        import httpx
        print("  ✅ httpx 已安装")
    except ImportError:
        print("  ❌ httpx 未安装")
    
    # 提供访问建议
    print("\n🌐 访问建议:")
    print("  管理界面: http://localhost:8081")
    print("  API端点: http://localhost:8080/v1/chat/completions")
    print("  健康检查: http://localhost:8081/api")
    
    # 检查防火墙（Windows）
    if os.name == 'nt':
        print("\n🔒 Windows防火墙检查:")
        try:
            result = subprocess.run(['netsh', 'advfirewall', 'show', 'allprofiles'], 
                                  capture_output=True, text=True)
            if "启用" in result.stdout or "Enabled" in result.stdout:
                print("  ⚠️  Windows防火墙已启用，请确保允许Python访问网络")
            else:
                print("  ✅ Windows防火墙未启用")
        except:
            print("  ❓ 无法检查防火墙状态")
    
    print("\n" + "=" * 50)
    print("💡 如果无法访问管理界面，请尝试:")
    print("1. 检查端口是否被占用")
    print("2. 确保所有依赖已安装")
    print("3. 检查防火墙设置")
    print("4. 尝试使用不同的端口")

if __name__ == "__main__":
    diagnose()