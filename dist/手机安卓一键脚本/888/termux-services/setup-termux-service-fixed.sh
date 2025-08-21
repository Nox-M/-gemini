#!/data/data/com.termux/files/usr/bin/bash
# LLM代理服务 - Termux服务安装和配置脚本（优化版）
# 解决兼容性问题并提供更好的错误处理

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 设置变量
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SERVICE_NAME="llm-proxy"
TERMUX_SERVICE_DIR="$HOME/.termux/service/$SERVICE_NAME"
LOG_DIR="$HOME/.llm-proxy/logs"
CONFIG_BACKUP="$HOME/.llm-proxy/config.backup"

# 打印带颜色的消息
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_banner() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════╗"
    echo "║      LLM代理服务 Termux优化安装       ║"
    echo "║        解决兼容性问题 v2.0            ║"
    echo "╚═══════════════════════════════════════╝"
    echo -e "${NC}"
}

# 检查是否在Termux环境中
check_termux() {
    if [[ "$PREFIX" != *"com.termux"* ]]; then
        print_error "此脚本只能在Termux环境中运行"
        exit 1
    fi
    
    # 检查是否在项目目录
    if [[ ! -f "$PROJECT_DIR/app.py" ]]; then
        print_error "请在项目根目录运行此脚本"
        exit 1
    fi
}

# 检查网络连接
check_network() {
    print_status "检查网络连接..."
    
    if ! curl -s --connect-timeout 5 https://pypi.org > /dev/null; then
        print_warning "网络连接可能有问题，安装依赖可能失败"
        print_warning "建议配置代理或使用国内镜像"
        
        # 配置pip国内镜像
        mkdir -p "$HOME/.pip"
        cat > "$HOME/.pip/pip.conf" << EOF
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF
        print_status "已配置pip使用清华镜像"
    else
        print_status "网络连接正常"
    fi
}

# 安装必要的Termux包
install_packages() {
    print_status "检查并安装必要的Termux包..."
    
    # 更新包列表
    pkg update -y
    
    # 安装必要的包
    local packages=("termux-services" "python" "curl" "procps" "net-tools" "openssl-tool")
    
    for pkg in "${packages[@]}"; do
        if ! dpkg -l | grep -q "^ii  $pkg "; then
            print_status "安装 $pkg..."
            if ! pkg install -y "$pkg"; then
                print_error "安装 $pkg 失败"
                exit 1
            fi
        else
            print_status "$pkg 已安装"
        fi
    done
    
    # 启用termux-services
    if [[ ! -d "$HOME/.termux/service" ]]; then
        mkdir -p "$HOME/.termux/service"
    fi
}

# 备份配置文件
backup_config() {
    if [[ -f "$PROJECT_DIR/config.ini" ]]; then
        mkdir -p "$(dirname "$CONFIG_BACKUP")"
        cp "$PROJECT_DIR/config.ini" "$CONFIG_BACKUP"
        print_status "已备份配置文件到: $CONFIG_BACKUP"
    fi
}

# 修复配置文件
fix_config() {
    print_status "检查并修复配置文件..."
    
    local config_file="$PROJECT_DIR/config.ini"
    
    # 如果配置文件不存在，创建默认配置
    if [[ ! -f "$config_file" ]]; then
        print_status "创建默认配置文件..."
        cat > "$config_file" << EOF
[SERVER]
port = 8080
host = 0.0.0.0
api_key = 123
min_response_length = 400
request_timeout = 180
web_port = 5001
web_host = 127.0.0.1

[API_KEYS]
group1 = ["YOUR_API_KEY_1", "YOUR_API_KEY_2", "YOUR_API_KEY_3", "YOUR_API_KEY_4"]
group2 = ["YOUR_API_KEY_5", "YOUR_API_KEY_6", "YOUR_API_KEY_7", "YOUR_API_KEY_8"]

[API]
base_url = https://generativelanguage.googleapis.com/v1beta
EOF
    fi
    
    # 检查并修复API密钥格式
    if grep -q "AIzaSyCgh-9h5PhprwiGSrk7oNxD5Bl240gI6Fk" "$config_file"; then
        print_warning "检测到示例API密钥，请替换为真实密钥"
        sed -i 's/AIzaSyCgh-9h5PhprwiGSrk7oNxD5Bl240gI6Fk/YOUR_API_KEY_1/g' "$config_file"
        sed -i 's/AIzaSyBmfY6uDjeDmaCbjjuDpMhLJe6H8nMMGXA/YOUR_API_KEY_2/g' "$config_file"
        sed -i 's/AIzaSyCRxaB09p2wEDJPbwc69tEukfrsv0HT5YQ/YOUR_API_KEY_3/g' "$config_file"
        sed -i 's/AIzaSyDJqNc2s-L2_RW0-AwMevHRvhYgEMMXLRM/YOUR_API_KEY_4/g' "$config_file"
        sed -i 's/AIzaSyDxG_Dn27XZ-OSeg_iWbGduohqD9gYrGiI/YOUR_API_KEY_5/g' "$config_file"
        sed -i 's/AIzaSyDP-WGwWX4SY2uLTaKAivWwuXzX0LqSui0/YOUR_API_KEY_6/g' "$config_file"
        sed -i 's/AIzaSyBwlIzbZ7bnRtYU7iicNdMnLYKkd8XVPDU/YOUR_API_KEY_7/g' "$config_file"
        sed -i 's/AIzaSyDIwwW4ApVM7Dsj7BuCq4766eCWcOW9_mM/YOUR_API_KEY_8/g' "$config_file"
    fi
}

# 设置termux-services
setup_termux_services() {
    print_status "配置termux-services..."
    
    # 创建服务目录
    mkdir -p "$TERMUX_SERVICE_DIR"
    
    # 使用优化版的服务脚本
    cp "$SCRIPT_DIR/llm-proxy-daemon-fixed" "$TERMUX_SERVICE_DIR/llm-proxy-daemon"
    chmod +x "$TERMUX_SERVICE_DIR/llm-proxy-daemon"
    
    # 复制服务配置
    cp "$SCRIPT_DIR/llm-proxy.service" "$HOME/.termux/service/"
    
    print_status "termux-services配置完成"
}

# 创建必要的目录
create_directories() {
    print_status "创建必要的目录..."
    
    mkdir -p "$LOG_DIR"
    mkdir -p "$HOME/.config/llm-proxy"
    mkdir -p "$HOME/.termux/boot"
    
    # 设置目录权限
    chmod 755 "$LOG_DIR"
    chmod 755 "$HOME/.config/llm-proxy"
    
    print_status "目录创建完成"
}

# 配置Python环境
setup_python_env() {
    print_status "配置Python环境..."
    
    # 升级pip
    python -m pip install --upgrade pip
    
    # 安装依赖
    local requirements_file="$PROJECT_DIR/termux-services/requirements-termux.txt"
    if [[ -f "$requirements_file" ]]; then
        print_status "安装Python依赖..."
        if pip install -r "$requirements_file"; then
            print_status "Python依赖安装完成"
        else
            print_error "Python依赖安装失败"
            print_error "请手动运行: pip install -r $requirements_file"
        fi
    else
        print_error "依赖文件不存在: $requirements_file"
        exit 1
    fi
}

# 设置开机自启
enable_boot_startup() {
    print_status "设置开机自启..."
    
    # 创建启动脚本
    cat > "$HOME/.termux/boot/llm-proxy-boot" << 'EOF'
#!/data/data/com.termux/files/usr/bin/bash
# LLM代理服务开机启动脚本

# 等待系统初始化
sleep 3

# 等待网络连接（可选）
# sleep 10

# 启动服务
sv up llm-proxy
EOF
    
    chmod +x "$HOME/.termux/boot/llm-proxy-boot"
    
    print_status "开机自启配置完成"
}

# 测试服务
test_service() {
    print_status "测试服务可用性..."
    
    # 等待服务启动
    sleep 5
    
    # 获取配置中的端口
    local api_port=$(grep -E "^port\s*=" "$PROJECT_DIR/config.ini" | cut -d'=' -f2 | tr -d ' ')
    
    if curl -s --connect-timeout 10 "http://localhost:$api_port/" > /dev/null 2>&1; then
        print_status "服务响应正常 ✓"
        print_status "API地址: http://localhost:$api_port"
    else
        print_warning "服务可能未完全启动，请等待几秒后重试"
        print_warning "查看日志: tail -f $LOG_DIR/llm-proxy.log"
    fi
}

# 显示使用说明
show_usage() {
    echo ""
    echo "======================================"
    echo "安装完成！"
    echo "======================================"
    echo ""
    echo "服务管理："
    echo "  启动: sv up llm-proxy"
    echo "  停止: sv down llm-proxy"
    echo "  重启: sv restart llm-proxy"
    echo "  状态: sv status llm-proxy"
    echo ""
    echo "日志管理："
    echo "  查看日志: $SCRIPT_DIR/log-manager.sh view"
    echo "  清理日志: $SCRIPT_DIR/log-manager.sh clean"
    echo ""
    echo "配置文件："
    echo "  主配置: $PROJECT_DIR/config.ini"
    echo "  日志目录: $LOG_DIR"
    echo ""
    echo "API测试："
    local api_port=$(grep -E "^port\s*=" "$PROJECT_DIR/config.ini" | cut -d'=' -f2 | tr -d ' ')
    echo "  curl -X POST http://localhost:$api_port/v1/chat/completions \\"
    echo "    -H \"Content-Type: application/json\" \\"
    echo "    -H \"Authorization: Bearer 123\" \\"
    echo "    -d '{\"model\":\"gemini-2.5-flash\",\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}]}'"
    echo ""
}

# 主程序
main() {
    print_banner
    
    check_termux
    check_network
    backup_config
    fix_config
    
    install_packages
    setup_termux_services
    create_directories
    setup_python_env
    enable_boot_startup
    
    # 询问是否立即启动服务
    echo ""
    read -p "是否立即启动服务？(y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "启动LLM代理服务..."
        sv up llm-proxy
        test_service
    fi
    
    show_usage
    
    print_status "LLM代理服务已成功配置为Termux服务！"
    print_status "服务将在设备重启后自动启动"
    print_status "如有问题，请查看日志文件"
}

# 运行主程序
main