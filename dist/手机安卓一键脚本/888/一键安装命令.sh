#!/data/data/com.termux/files/usr/bin/bash
# LLM代理服务 - 超级一键安装命令
# 复制粘贴即可运行的完整安装脚本

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

print_banner() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════╗"
    echo "║      LLM代理服务 一键安装器           ║"
    echo "║        复制粘贴即可运行               ║"
    echo "╚═══════════════════════════════════════╝"
    echo -e "${NC}"
}

# 主安装流程
main() {
    print_banner
    
    echo -e "${GREEN}开始一键安装LLM代理服务...${NC}"
    
    # 1. 更新Termux
    echo "更新Termux..."
    pkg update -y && pkg upgrade -y
    
    # 2. 安装必要工具
    echo "安装必要工具..."
    pkg install git curl wget -y
    
    # 3. 克隆项目
    echo "下载项目..."
    cd ~
    if [ -d "LLM代理服务_Web版" ]; then
        echo "项目已存在，更新中..."
        cd LLM代理服务_Web版
        git pull || echo "无法更新，使用现有版本"
    else
        echo "克隆项目..."
        # 这里替换为实际的git仓库地址
        git clone https://github.com/your-username/LLM代理服务_Web版.git || {
            echo "使用备用下载方式..."
            # 备用下载方式
            curl -L -o llm-proxy.zip "https://github.com/your-username/LLM代理服务_Web版/archive/main.zip"
            unzip llm-proxy.zip
            mv "LLM代理服务_Web版-main" "LLM代理服务_Web版"
            rm llm-proxy.zip
        }
        cd LLM代理服务_Web版
    fi
    
    # 4. 运行安装脚本
    echo "运行安装脚本..."
    if [ -f "install-termux-fixed.sh" ]; then
        chmod +x install-termux-fixed.sh
        ./install-termux-fixed.sh
    else
        chmod +x install-termux.sh
        ./install-termux.sh
    fi
    
    echo -e "${GREEN}安装完成！${NC}"
}

# 运行主程序
main