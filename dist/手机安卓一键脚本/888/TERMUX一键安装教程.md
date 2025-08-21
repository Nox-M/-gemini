# LLM代理服务 - Termux一键安装教程

## 📱 概述
本教程将帮助你在安卓Termux环境中一键安装并运行LLM代理服务，解决所有已知的兼容性问题。

## 🚀 一键安装命令

### 方法1：直接克隆安装（推荐）
```bash
# 1. 安装Termux（如果尚未安装）
# 从F-Droid或GitHub下载最新版Termux

# 2. 更新Termux并安装必要工具
pkg update && pkg upgrade -y
pkg install git curl wget -y

# 3. 克隆项目
cd ~
git clone https://github.com/your-repo/LLM代理服务_Web版.git
cd LLM代理服务_Web版

# 4. 一键安装
chmod +x install-termux-fixed.sh
./install-termux-fixed.sh
```

### 方法2：手动下载安装
```bash
# 1. 下载项目文件到手机
# 2. 在Termux中导航到项目目录
cd /sdcard/Download/LLM代理服务_Web版  # 或你的实际路径

# 3. 一键安装
chmod +x install-termux-fixed.sh
./install-termux-fixed.sh
```

## 📋 安装前准备

### 必需条件
- ✅ Android 7.0+
- ✅ Termux最新版（建议从F-Droid安装）
- ✅ 至少200MB存储空间
- ✅ 网络连接（用于下载依赖）

### 权限设置
```bash
# 首次运行前执行
termux-setup-storage
```

## 🔧 安装步骤详解

### 第1步：环境检查
安装脚本会自动检查：
- Termux环境完整性
- 网络连接状态
- 存储权限
- 必要软件包

### 第2步：依赖安装
自动安装：
- Python 3.x
- termux-services（进程管理）
- 网络工具（net-tools, curl）
- Python依赖包（已针对Termux优化）

### 第3步：服务配置
- 自动配置termux-services
- 设置开机自启
- 创建日志目录
- 配置端口自动检测

### 第4步：服务启动
- 自动启动LLM代理服务
- 测试服务可用性
- 显示访问信息

## 🎯 使用方法

### 服务管理命令
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

### 日志查看
```bash
# 实时查看日志
tail -f ~/.llm-proxy/logs/llm-proxy.log

# 查看最近50行
./termux-services/log-manager.sh view

# 清理日志
./termux-services/log-manager.sh clean
```

### 配置修改
```bash
# 编辑配置文件
nano config.ini

# 重启服务使配置生效
sv restart llm-proxy
```

## ⚙️ 配置文件说明

### config.ini 关键配置
```ini
[SERVER]
port = 8080              # API端口（自动检测冲突）
host = 0.0.0.0          # 监听地址
api_key = 123           # 访问密钥（请修改）
min_response_length = 400  # 最小响应长度
request_timeout = 180   # 请求超时时间

[API_KEYS]
group1 = ["YOUR_API_KEY_1", "YOUR_API_KEY_2"]  # 替换为真实API密钥
group2 = ["YOUR_API_KEY_3", "YOUR_API_KEY_4"]

[API]
base_url = https://generativelanguage.googleapis.com/v1beta
```

## 🔍 故障排除

### 常见问题及解决方案

#### 1. 依赖安装失败
```bash
# 手动安装依赖
pip install -r termux-services/requirements-termux.txt --no-cache-dir
```

#### 2. 端口被占用
- 安装脚本会自动检测并调整端口
- 手动修改config.ini中的端口号

#### 3. 网络连接问题
```bash
# 配置国内镜像
mkdir -p ~/.pip
echo "[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn" > ~/.pip/pip.conf
```

#### 4. API调用失败
- 检查API密钥是否正确
- 确认网络可以访问Google API
- 考虑使用代理或国内API镜像

### 调试命令
```bash
# 检查服务状态
sv status llm-proxy

# 查看详细日志
cat ~/.llm-proxy/logs/llm-proxy.log

# 手动测试
curl http://localhost:8080/
```

## 📊 性能优化建议

### 内存优化
- 关闭不必要的Termux会话
- 使用`sv down llm-proxy`停止服务时释放内存

### 电池优化
- 服务在后台运行时功耗极低
- 不需要时可完全停止服务

## 🔄 更新升级

### 更新项目
```bash
# 停止服务
sv down llm-proxy

# 更新代码
git pull

# 重新安装
./install-termux-fixed.sh
```

### 重新安装
```bash
# 完全卸载
sv down llm-proxy
rm -rf ~/.termux/service/llm-proxy
rm -rf ~/.llm-proxy

# 重新安装
./install-termux-fixed.sh
```

## 📞 技术支持

### 获取帮助
1. 查看日志文件：`~/.llm-proxy/logs/llm-proxy.log`
2. 运行诊断命令：`./termux-services/log-manager.sh view`
3. 重启服务：`sv restart llm-proxy`

### 联系支持
- 提交Issue到项目GitHub
- 查看项目文档和FAQ

## ✅ 安装验证

安装成功后，你应该看到：
```
======================================
安装完成！
======================================

服务管理：
  启动: sv up llm-proxy
  停止: sv down llm-proxy
  重启: sv restart llm-proxy
  状态: sv status llm-proxy

API端点：
  地址: http://localhost:8080
  测试: curl -X POST http://localhost:8080/v1/chat/completions
```

## 🎉 恭喜！
现在你可以在安卓手机上使用LLM代理服务了！服务已配置为开机自启，无需每次手动启动。