#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试统计功能的简单脚本
"""

import requests
import json
import time

def test_stats_api():
    """测试统计API"""
    base_url = "http://localhost:5000"
    
    print("=== 测试统计功能 ===")
    
    # 1. 获取初始统计
    try:
        response = requests.get(f"{base_url}/api/stats")
        if response.status_code == 200:
            stats = response.json()
            print("✓ 统计API正常")
            print(f"初始统计: {json.dumps(stats, indent=2, ensure_ascii=False)}")
        else:
            print(f"✗ 统计API错误: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ 连接失败: {e}")
        return False
    
    # 2. 重置统计
    try:
        response = requests.post(f"{base_url}/api/stats/reset")
        if response.status_code == 200:
            result = response.json()
            print("✓ 重置统计成功")
        else:
            print(f"✗ 重置统计失败: {response.status_code}")
    except Exception as e:
        print(f"✗ 重置统计失败: {e}")
    
    # 3. 发送测试请求
    print("\n=== 发送测试请求 ===")
    
    test_payload = {
        "model": "gemini-2.5-flash",
        "messages": [
            {"role": "user", "content": "你好，这是一个测试请求"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    headers = {
        "Authorization": "Bearer 123",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=test_payload,
            headers=headers
        )
        print(f"测试请求状态码: {response.status_code}")
        if response.status_code == 200:
            print("✓ 测试请求成功")
        else:
            print(f"测试请求响应: {response.text}")
    except Exception as e:
        print(f"✗ 测试请求失败: {e}")
    
    # 4. 再次获取统计
    time.sleep(2)  # 等待统计更新
    
    try:
        response = requests.get(f"{base_url}/api/stats")
        if response.status_code == 200:
            stats = response.json()
            print("\n=== 更新后的统计 ===")
            print(json.dumps(stats, indent=2, ensure_ascii=False))
            
            # 验证统计是否正确
            if stats['stats']['total_requests'] > 0:
                print("✓ 统计已正确记录")
                return True
            else:
                print("✗ 统计未正确记录")
                return False
    except Exception as e:
        print(f"✗ 获取更新统计失败: {e}")
        return False

if __name__ == "__main__":
    print("等待服务启动...")
    time.sleep(3)
    test_stats_api()