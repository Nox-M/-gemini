#!/data/data/com.termux/files/usr/bin/bash
# LLM代理服务 - Termux服务安装和配置脚本
# 自动配置termux-services并设置开机自启

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 设置变量
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SERVICE_NAME="llm-proxy"
TERMUX_SERVICE_DIR="$HOME/.termux/service/$SERVICE_NAME"
LOG_DIR="$HOME/.llm-proxy/logs"

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

# 检查是否在Termux环境中
check_termux() {
    if [[ "$PREFIX" != *"com.termux"* ]]; then
        print_error "此脚本只能在Termux环境中运行"
        exit 1
    fi
}

# 安装必要的Termux包
install_packages() {
    print_status "检查并安装必要的Termux包..."
    
    # 更新包列表
    pkg update -y
    
    # 安装必要的包
    local packages=("termux-services" "python" "curl" "procps" "net-tools")
    
    for pkg in "${packages[@]}"; do
        if ! dpkg -l | grep -q "^ii  $pkg "; then
            print_status "安装 $pkg..."
            pkg install -y "$pkg"
        else
            print_status "$pkg 已安装"
        fi
    done
}

# 设置termux-services
setup_termux_services() {
    print_status "配置termux-services..."
    
    # 启用termux-services
    if [[ ! -d "$HOME/.termux/service" ]]; then
        mkdir -p "$HOME/.termux/service"
    fi
    
    # 创建服务目录
    mkdir -p "$TERMUX_SERVICE_DIR"
    
    # 复制服务文件
    cp "$SCRIPT_DIR/llm-proxy-daemon" "$TERMUX_SERVICE_DIR/"
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
    
    print_status "目录创建完成"
}

# 配置Python环境
setup_python_env() {
    print_status "配置Python环境..."
    
    # 升级pip
    python -m pip install --upgrade pip
    
    # 安装依赖
    if [[ -f "$PROJECT_DIR/requirements.txt" ]]; then
        print_status "安装Python依赖..."
        pip install -r "$PROJECT_DIR/requirements.txt"
    else
        print_warning "requirements.txt 文件不存在，跳过依赖安装"
    fi
}

# 设置开机自启
enable_boot_startup() {
    print_status "设置开机自启..."
    
    # 创建启动脚本
    cat > "$HOME/.termux/boot/llm-proxy-boot" << 'EOF'
#!/data/data/com.termux/files/usr/bin/bash
# LLM代理服务开机启动脚本

# 等待网络连接
sleep 5

# 启动服务
sv up llm-proxy
EOF
    
    chmod +x "$HOME/.termux/boot/llm-proxy-boot"
    
    # 创建boot目录（如果不存在）
    mkdir -p "$HOME/.termux/boot"
    
    print_status "开机自启配置完成"
}

# 启动服务
start_service() {
    print_status "启动LLM代理服务..."
    
    # 启动服务
    sv up llm-proxy
    
    # 等待服务启动
    sleep 3
    
    # 检查服务状态
    if sv status llm-proxy | grep -q "run:"; then
        print_status "服务启动成功！"
    else
        print_warning "服务启动可能失败，请检查日志"
    fi
}

# 显示服务状态
show_service_status() {
    print_status "服务状态信息："
    echo "--------------------------------"
    sv status llm-proxy
    echo "--------------------------------"
    
    # 显示日志位置
    echo "日志文件位置："
    echo "  主日志: $LOG_DIR/llm-proxy.log"
    echo "  错误日志: $LOG_DIR/llm-proxy-error.log"
    echo ""
    echo "查看日志命令："
    echo "  查看最近50行: $SCRIPT_DIR/log-manager.sh view"
    echo "  实时查看: tail -f $LOG_DIR/llm-proxy.log"
    echo ""
    echo "服务管理命令："
    echo "  启动: sv up llm-proxy"
    echo "  停止: sv down llm-proxy"
    echo "  重启: sv restart llm-proxy"
    echo "  状态: sv status llm-proxy"
}

# 主程序
main() {
    print_status "开始配置LLM代理服务..."
    
    check_termux
    install_packages
    setup_termux_services
    create_directories
    setup_python_env
    enable_boot_startup
    
    # 询问是否立即启动服务
    read -p "是否立即启动服务？(y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        start_service
    fi
    
    show_service_status
    
    print_status "配置完成！"
}

# 运行主程序
main