# LLM代理服务 - Termux版

## 📱 项目概述
这是一个专为Android Termux环境优化的LLM代理服务，使用Python + FastAPI + httpx构建。服务可以在安卓手机上运行，提供OpenAI兼容的API接口。

## ✨ 主要特性
- 🚀 **Termux优化**: 专为Android Termux环境深度优化
- 🌐 **Web管理界面**: 移动端友好的Web管理界面
- 🔧 **一键安装**: 提供完整的Termux一键安装脚本
- 📊 **服务管理**: 完整的服务启动、停止、重启功能
- 📝 **日志管理**: 自动日志轮转和管理
- 🔄 **自动恢复**: 服务崩溃后自动重启
- 🛡️ **API密钥管理**: 支持多组API密钥轮询使用

## 📋 系统要求
- Android 7.0+
- Termux (建议从F-Droid安装)
- 至少200MB存储空间
- 网络连接

## 🚀 快速开始

### 1. 安装Termux
从F-Droid或GitHub下载并安装Termux应用。

### 2. 一键安装
```bash
# 更新Termux
pkg update && pkg upgrade -y

# 安装必要工具
pkg install git curl wget -y

# 克隆项目
cd ~
git clone https://github.com/your-repo/LLM代理服务_Web版.git
cd LLM代理服务_Web版

# 一键安装
chmod +x install-termux-fixed.sh
./install-termux-fixed.sh
```

### 3. 服务管理
```bash
# 启动服务
sv up llm-proxy

# 停止服务
sv down llm-proxy

# 重启服务
sv restart llm-proxy

# 查看状态
sv status llm-proxy
```

## ⚙️ 配置说明

### 配置文件位置
- 主配置文件: `config.ini`
- 日志目录: `~/.llm-proxy/logs/`

### 主要配置项
```ini
[SERVER]
port = 8080              # 服务端口
host = 0.0.0.0          # 监听地址
api_key = 123           # API访问密钥
min_response_length = 400  # 最小响应长度
request_timeout = 180   # 请求超时时间

[API_KEYS]
group1 = ["YOUR_API_KEY_1", "YOUR_API_KEY_2"]
group2 = ["YOUR_API_KEY_3", "YOUR_API_KEY_4"]

[API]
base_url = https://generativelanguage.googleapis.com/v1beta
```

## 🌐 API使用

### API端点
- **聊天完成**: `POST /v1/chat/completions`
- **Web管理界面**: `GET /`
- **服务状态**: `GET /api`

### 请求示例
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 123" \
  -d '{
    "model": "gemini-pro",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": false
  }'
```

## 📱 Web管理界面
访问 `http://localhost:8080` 打开Web管理界面，提供：
- 服务状态监控
- 配置文件编辑
- API密钥管理
- 日志查看
- 快速启动/停止服务

## 🔧 故障排除

### 常见问题
1. **依赖安装失败**
   ```bash
   pip install -r requirements.txt --no-cache-dir
   ```

2. **端口被占用**
   - 修改`config.ini`中的端口号
   - 或使用自动端口检测功能

3. **服务无法启动**
   ```bash
   # 查看日志
   tail -f ~/.llm-proxy/logs/llm-proxy.log
   
   # 手动测试
   python app.py
   ```

4. **网络连接问题**
   ```bash
   # 配置国内镜像
   mkdir -p ~/.pip
   echo "[global]
   index-url = https://pypi.tuna.tsinghua.edu.cn/simple
   trusted-host = pypi.tuna.tsinghua.edu.cn" > ~/.pip/pip.conf
   ```

## 📊 日志管理
```bash
# 查看日志
./termux-services/log-manager.sh view

# 清理日志
./termux-services/log-manager.sh clean

# 查看服务状态
./termux-services/log-manager.sh status
```

## 🔄 更新升级
```bash
# 停止服务
sv down llm-proxy

# 更新代码
git pull

# 重新安装
./install-termux-fixed.sh
```

## 📞 技术支持
- 查看日志文件: `~/.llm-proxy/logs/llm-proxy.log`
- 运行诊断: `./termux-services/log-manager.sh view`
- 重启服务: `sv restart llm-proxy`

## 📄 许可证
本项目采用MIT许可证。

## 🤝 贡献
欢迎提交Issue和Pull Request！

---

**注意**: 请确保在使用前替换配置文件中的API密钥为您的真实密钥。