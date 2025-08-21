#!/data/data/com.termux/files/usr/bin/bash
# LLM代理服务 - Termux一键安装脚本（优化版）
# 解决所有已知兼容性问题并提供完整错误处理

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 打印函数
print_banner() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════╗"
    echo "║      LLM代理服务 Termux优化安装       ║"
    echo "║        一键解决兼容性问题             ║"
    echo "╚═══════════════════════════════════════╝"
    echo -e "${NC}"
}

print_step() {
    echo -e "${GREEN}[步骤 $1/$2]${NC} $3"
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

# 检查环境
check_environment() {
    if [[ "$PREFIX" != *"com.termux"* ]]; then
        print_error "此脚本只能在Termux环境中运行"
        exit 1
    fi
    
    if [[ ! -f "app.py" ]]; then
        print_error "请在项目根目录运行此脚本"
        exit 1
    fi
}

# 检查存储权限
check_storage_permission() {
    print_info "检查存储权限..."
    
    if [[ ! -w "$HOME" ]]; then
        print_error "没有写入权限，请运行: termux-setup-storage"
        exit 1
    fi
    
    # 创建测试文件
    touch "$HOME/.termux-permission-test" 2>/dev/null && rm "$HOME/.termux-permission-test"
    if [[ $? -ne 0 ]]; then
        print_error "存储权限不足，请运行: termux-setup-storage"
        exit 1
    fi
}

# 主安装流程
main() {
    print_banner
    
    check_environment
    check_storage_permission
    
    TOTAL_STEPS=7
    CURRENT_STEP=0
    
    # 步骤1：检查并安装依赖
    ((CURRENT_STEP++))
    print_step $CURRENT_STEP $TOTAL_STEPS "检查并安装Termux依赖包"
    if [[ -d "termux-services" ]]; then
        cd termux-services
        if [[ -f "setup-termux-service-fixed.sh" ]]; then
            bash setup-termux-service-fixed.sh
        else
            bash setup-termux-service.sh
        fi
    else
        print_error "termux-services目录不存在，请确保项目完整"
        exit 1
    fi
    
    # 步骤2：验证安装
    ((CURRENT_STEP++))
    print_step $CURRENT_STEP $TOTAL_STEPS "验证服务安装"
    
    if command -v sv &> /dev/null; then
        print_info "termux-services已正确安装"
    else
        print_error "termux-services安装失败"
        exit 1
    fi
    
    # 步骤3：检查服务状态
    ((CURRENT_STEP++))
    print_step $CURRENT_STEP $TOTAL_STEPS "检查服务状态"
    
    sleep 2
    if sv status llm-proxy | grep -q "run:"; then
        print_info "服务已成功启动"
    else
        print_warning "服务可能未启动，请检查日志"
    fi
    
    # 步骤4：显示访问信息
    ((CURRENT_STEP++))
    print_step $CURRENT_STEP $TOTAL_STEPS "显示服务信息"
    
    # 获取实际使用的端口
    local api_port=$(grep -E "^port\s*=" config.ini 2>/dev/null | cut -d'=' -f2 | tr -d ' ' || echo "8080")
    
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
    echo "  查看日志: ./termux-services/log-manager.sh view"
    echo "  清理日志: ./termux-services/log-manager.sh clean"
    echo ""
    echo "配置文件："
    echo "  主配置: $(pwd)/config.ini"
    echo "  日志目录: ~/.llm-proxy/logs/"
    echo ""
    echo "API端点："
    echo "  地址: http://localhost:$api_port"
    echo "  测试: curl -X POST http://localhost:$api_port/v1/chat/completions"
    echo ""
    
    # 步骤5：测试服务
    ((CURRENT_STEP++))
    print_step $CURRENT_STEP $TOTAL_STEPS "测试服务可用性"
    
    sleep 3
    if curl -s "http://localhost:$api_port/" > /dev/null 2>&1; then
        print_info "服务响应正常 ✓"
    else
        print_warning "服务可能未完全启动，请等待几秒后重试"
    fi
    
    # 步骤6：显示配置提示
    ((CURRENT_STEP++))
    print_step $CURRENT_STEP $TOTAL_STEPS "配置提示"
    
    echo ""
    echo "重要提醒："
    echo "1. 请编辑 config.ini 文件，替换 YOUR_API_KEY_X 为真实的API密钥"
    echo "2. 如果遇到网络问题，请配置代理或使用国内API镜像"
    echo "3. 查看日志命令: tail -f ~/.llm-proxy/logs/llm-proxy.log"
    echo ""
    
    # 步骤7：完成提示
    ((CURRENT_STEP++))
    print_step $CURRENT_STEP $TOTAL_STEPS "安装完成"
    
    echo ""
    print_info "LLM代理服务已成功配置为Termux服务！"
    print_info "服务将在设备重启后自动启动"
    print_info "如有问题，请查看日志文件"
}

# 错误处理
trap 'print_error "脚本执行过程中出现错误，请检查上面的错误信息"; exit 1' ERR

# 运行主程序
main