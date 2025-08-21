#!/bin/bash
# 将优化后的LLM代理服务更新到GitHub项目

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════╗"
echo "║      更新GitHub项目                   ║"
echo "║    LLM代理服务Termux优化版            ║"
echo "╚═══════════════════════════════════════╝"
echo -e "${NC}"

# 检查是否在项目目录
if [ ! -f "app.py" ]; then
    echo "错误：请在LLM代理服务_Web版目录中运行此脚本"
    exit 1
fi

# 创建GitHub项目结构
echo -e "${GREEN}创建GitHub项目结构...${NC}"

# 创建README.md
cat > README.md << 'EOF'
# LLM代理服务 - Termux优化版

## 🚀 一键在安卓Termux上运行LLM代理服务

### 📱 特性
- ✅ 专为Termux环境优化
- ✅ 一键安装，零配置
- ✅ 自动处理兼容性问题
- ✅ 支持开机自启
- ✅ 完整的错误处理

### 🔧 一键安装

**复制粘贴即可运行：**
```bash
pkg update -y && pkg upgrade -y
pkg install git curl -y
cd ~
git clone https://github.com/adc666sav466/-gemini-.git
cd -gemini-
chmod +x install-termux-fixed.sh
./install-termux-fixed.sh
```

### 📋 使用方法

安装完成后：
```bash
# 启动服务
sv up llm-proxy

# 查看状态
sv status llm-proxy

# 查看日志
tail -f ~/.llm-proxy/logs/llm-proxy.log
```

### ⚙️ 配置

编辑 `config.ini` 文件，替换API密钥为你的真实密钥。

### 📖 详细文档
查看 [TERMUX一键安装教程.md](TERMUX一键安装教程.md) 获取完整指南。

## 🎯 项目结构
```
LLM代理服务_Web版/
├── app.py                    # 主程序
├── config.ini               # 配置文件
├── requirements.txt         # 依赖列表
├── install-termux-fixed.sh  # 一键安装脚本
├── 一键安装命令.sh          # 超级一键安装
├── TERMUX一键安装教程.md    # 详细教程
├── termux-services/         # Termux服务配置
│   ├── requirements-termux.txt  # Termux优化依赖
│   ├── setup-termux-service-fixed.sh  # 服务安装脚本
│   ├── llm-proxy-daemon-fixed        # 优化守护进程
│   ├── llm-proxy.service            # 服务配置
│   └── log-manager.sh              # 日志管理
├── static/                  # 静态文件
├── templates/               # 模板文件
└── README.md               # 项目说明
```

## 🔍 故障排除
- 查看日志：`tail -f ~/.llm-proxy/logs/llm-proxy.log`
- 重启服务：`sv restart llm-proxy`
- 检查配置：确保config.ini中的API密钥正确

## 📞 支持
如有问题，请提交Issue或查看详细教程。
EOF

# 复制优化文件到正确位置
echo -e "${GREEN}复制优化文件...${NC}"

# 确保termux-services目录存在
mkdir -p termux-services

# 设置文件权限
chmod +x install-termux-fixed.sh
chmod +x 一键安装命令.sh
chmod +x termux-services/setup-termux-service-fixed.sh
chmod +x termux-services/llm-proxy-daemon-fixed
chmod +x termux-services/log-manager.sh

# 创建.gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# 日志
*.log
logs/
.llm-proxy/

# 临时文件
*.tmp
*.bak
.DS_Store
Thumbs.db

# 配置文件（可选）
# config.ini
EOF

# 显示完成信息
echo -e "${GREEN}✅ 项目已准备好上传到GitHub！${NC}"
echo ""
echo "下一步："
echo "1. cd 到你的GitHub项目目录"
echo "2. 复制这些文件到该目录"
echo "3. 运行："
echo "   git add ."
echo "   git commit -m 'Add Termux optimized LLM proxy service'"
echo "   git push origin main"
echo ""
echo "或者直接使用："
echo "   git clone https://github.com/adc666sav466/-gemini-.git"
echo "   然后替换文件并推送"