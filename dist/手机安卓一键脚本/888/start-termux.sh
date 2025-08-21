#!/data/data/com.termux/files/usr/bin/bash
# LLM代理服务 - Termux启动脚本
# 优化后的启动脚本，支持多种启动模式

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 设置变量
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
SERVICE_NAME="llm-proxy"
LOG_DIR="$HOME/.llm-proxy/logs"
PID_FILE="$LOG_DIR/llm-proxy.pid"

# 打印函数
print_banner() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    LLM代理服务 - Termux版                      ║"
    echo "║                      优化启动脚本                              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# 检查环境
check_environment() {
    print_step "检查运行环境..."
    
    if [[ "$PREFIX" != *"com.termux"* ]]; then
        print_error "此脚本只能在Termux环境中运行"
        exit 1
    fi
    
    if [[ ! -f "$PROJECT_DIR/app.py" ]]; then
        print_error "未找到app.py文件，请在项目根目录运行此脚本"
        exit 1
    fi
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python未安装，请运行: pkg install python3"
        exit 1
    fi
    
    print_info "环境检查通过"
}

# 创建必要目录
create_directories() {
    print_step "创建必要目录..."
    
    mkdir -p "$LOG_DIR"
    mkdir -p "$HOME/.config/llm-proxy"
    
    print_info "目录创建完成"
}

# 检查并安装依赖
check_dependencies() {
    print_step "检查Python依赖..."
    
    # 检查是否有requirements.txt
    if [[ ! -f "$PROJECT_DIR/requirements.txt" ]]; then
        print_warning "requirements.txt文件不存在，跳过依赖检查"
        return
    fi
    
    # 检查依赖是否已安装
    if ! /data/data/com.termux/files/usr/bin/python3 -c "import fastapi" 2>/dev/null; then
        print_info "正在安装Python依赖..."
        /data/data/com.termux/files/usr/bin/python3 -m pip install -r "$PROJECT_DIR/requirements.txt" --no-cache-dir
        if [[ $? -eq 0 ]]; then
            print_info "依赖安装完成"
        else
            print_error "依赖安装失败"
            exit 1
        fi
    else
        print_info "Python依赖已安装"
    fi
}

# 检查端口是否可用
check_port() {
    local port=$1
    if netstat -tuln 2>/dev/null | grep -q ":$port "; then
        print_warning "端口 $port 已被占用"
        return 1
    fi
    return 0
}

# 获取配置端口
get_config_port() {
    local config_file="$PROJECT_DIR/config.ini"
    if [[ -f "$config_file" ]]; then
        grep -E "^port.*=" "$config_file" | head -1 | sed 's/.*= *//' | tr -d ' '
    else
        echo "5000"
    fi
}

# 启动服务
start_service() {
    print_step "启动LLM代理服务..."
    
    # 检查服务是否已在运行
    if [[ -f "$PID_FILE" ]]; then
        local old_pid=$(cat "$PID_FILE")
        if kill -0 "$old_pid" 2>/dev/null; then
            print_warning "服务已在运行 (PID: $old_pid)"
            return 0
        else
            print_info "清理旧的PID文件"
            rm -f "$PID_FILE"
        fi
    fi
    
    # 获取配置端口（统一端口配置）
    local port=$(get_config_port)
    
    # 检查端口
    if ! check_port "$port"; then
        print_error "端口 $port 被占用，请修改config.ini中的端口配置"
        exit 1
    fi
    
    # 设置环境变量
    export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
    export PYTHONIOENCODING='utf-8'
    export PYTHONUNBUFFERED='1'
    
    # 启动服务
    cd "$PROJECT_DIR"
    
    print_info "启动服务中..."
    print_info "HTTP状态码将同时显示在终端和日志中..."
    
    # 创建日志目录
    mkdir -p "$LOG_DIR"
    
    # 启动服务并同时输出到终端和日志文件
    /data/data/com.termux/files/usr/bin/python3 app.py 2>&1 | tee "$LOG_DIR/llm-proxy.log" &
    
    local pid=$!
    echo $pid > "$PID_FILE"
    
    # 等待服务启动
    sleep 3
    
    # 检查服务是否成功启动
    if kill -0 "$pid" 2>/dev/null; then
        print_info "服务启动成功！"
        print_info "PID: $pid"
        print_info "端口: $port"
        print_info "日志: $LOG_DIR/llm-proxy.log"
        print_info "访问地址: http://localhost:$port"
        print_info "管理界面: http://localhost:$port/"
        print_info "API端点: http://localhost:$port/v1/chat/completions"
        
        # 测试服务响应并显示HTTP状态码
        local response=$(curl -s -w "HTTP状态码: %{http_code}" "http://localhost:$port/" 2>/dev/null)
        if [[ $? -eq 0 ]]; then
            print_info "服务响应测试通过 ✓"
            print_info "$response"
        else
            print_warning "服务响应测试失败，请检查日志"
        fi
    else
        print_error "服务启动失败，请检查日志: $LOG_DIR/llm-proxy.log"
        exit 1
    fi
}

# 停止服务
stop_service() {
    print_step "停止LLM代理服务..."
    
    if [[ -f "$PID_FILE" ]]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            print_info "停止服务 (PID: $pid)..."
            kill "$pid"
            
            # 等待进程结束
            local count=0
            while kill -0 "$pid" 2>/dev/null && [[ $count -lt 10 ]]; do
                sleep 1
                ((count++))
            done
            
            if kill -0 "$pid" 2>/dev/null; then
                print_warning "正常停止失败，强制终止..."
                kill -9 "$pid"
            fi
            
            rm -f "$PID_FILE"
            print_info "服务已停止"
        else
            print_info "服务未运行"
            rm -f "$PID_FILE"
        fi
    else
        print_info "PID文件不存在，服务可能未运行"
    fi
}

# 重启服务
restart_service() {
    print_step "重启LLM代理服务..."
    stop_service
    sleep 2
    start_service
}

# 查看服务状态
show_status() {
    print_step "服务状态..."
    
    if [[ -f "$PID_FILE" ]]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            print_info "服务状态: 运行中"
            print_info "PID: $pid"
            
            # 显示端口信息
            local port=$(get_config_port)
            if netstat -tuln 2>/dev/null | grep -q ":$port "; then
                print_info "端口: $port (监听中)"
            else
                print_warning "端口: $port (未监听)"
            fi
            
            # 显示内存使用
            if command -v ps &> /dev/null; then
                local mem_info=$(ps -p "$pid" -o rss= 2>/dev/null | tr -d ' ')
                if [[ -n "$mem_info" ]]; then
                    local mem_mb=$((mem_info / 1024))
                    print_info "内存使用: ${mem_mb}MB"
                fi
            fi
            
            # 显示运行时间
            if command -v ps &> /dev/null; then
                local etime=$(ps -p "$pid" -o etime= 2>/dev/null | tr -d ' ')
                if [[ -n "$etime" ]]; then
                    print_info "运行时间: $etime"
                fi
            fi
        else
            print_info "服务状态: 未运行 (PID文件存在但进程不存在)"
            rm -f "$PID_FILE"
        fi
    else
        print_info "服务状态: 未运行"
    fi
}

# 查看日志
show_logs() {
    local lines=${1:-50}
    
    if [[ -f "$LOG_DIR/llm-proxy.log" ]]; then
        print_info "显示最近 $lines 行日志:"
        echo "----------------------------------------"
        tail -n "$lines" "$LOG_DIR/llm-proxy.log"
        echo "----------------------------------------"
    else
        print_info "日志文件不存在"
    fi
}

# 显示帮助信息
show_help() {
    echo "LLM反截断代理服务 - Termux启动脚本"
    echo
    echo "用法: $0 [命令]"
    echo
    echo "命令:"
    echo "   chmod +x start-termux.sh         首先运行这个"
    echo "   ./start-termux.sh start       然后再运行这个"
    echo "  https://github.com/adc666sav466/-gemini-"
    echo "  若对您有帮助的话，github点点star"
    echo "  欢迎加入QQ群  1033083986"
    echo "  install        安装依赖"
    echo "  help           显示此帮助信息"
    echo
    echo "示例:"
    echo "  $0 start                    # 启动服务"
    echo "  $0 logs 100                # 查看最近100行日志"
    echo "  $0 status                  # 查看服务状态"
    echo
}

# 安装依赖
install_deps() {
    print_step "安装依赖..."
    check_environment
    create_directories
    check_dependencies
    print_info "依赖安装完成"
}

# 主程序
main() {
    case "${1:-help}" in
        "start")
            print_banner
            check_environment
            create_directories
            check_dependencies
            start_service
            ;;
        "stop")
            print_banner
            stop_service
            ;;
        "restart")
            print_banner
            restart_service
            ;;
        "status")
            print_banner
            show_status
            ;;
        "logs")
            print_banner
            show_logs "${2:-50}"
            ;;
        "install")
            print_banner
            install_deps
            ;;
        "help"|*)
            print_banner
            show_help
            ;;
    esac
}

# 运行主程序
main "$@"