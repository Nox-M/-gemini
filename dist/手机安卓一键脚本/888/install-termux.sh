#!/data/data/com.termux/files/usr/bin/bash
# LLM代理服务 - Termux一键安装脚本
# 将整个项目配置为Termux服务的完整解决方案

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
    echo "║      LLM代理服务 Termux安装器         ║"
    echo "║        一键配置为系统服务             ║"
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

# 主安装流程
main() {
    print_banner
    
    check_environment
    
    TOTAL_STEPS=6
    CURRENT_STEP=0
    
    # 步骤1：检查并安装依赖
    ((CURRENT_STEP++))
    print_step $CURRENT_STEP $TOTAL_STEPS "检查并安装Termux依赖包"
    if [[ -d "termux-services" ]]; then
        cd termux-services
        bash setup-termux-service.sh
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
    echo "  地址: http://localhost:8080"
    echo "  测试: curl -X POST http://localhost:8080/v1/chat/completions"
    echo ""
    
    # 步骤5：测试服务
    ((CURRENT_STEP++))
    print_step $CURRENT_STEP $TOTAL_STEPS "测试服务可用性"
    
    sleep 3
    if curl -s http://localhost:8080/ > /dev/null; then
        print_info "服务响应正常 ✓"
    else
        print_warning "服务可能未完全启动，请等待几秒后重试"
    fi
    
    # 步骤6：完成提示
    ((CURRENT_STEP++))
    print_step $CURRENT_STEP $TOTAL_STEPS "安装完成"
    
    echo ""
    print_info "LLM代理服务已成功配置为Termux服务！"
    print_info "服务将在设备重启后自动启动"
    print_info "如有问题，请查看日志文件"
}

# 运行主程序
main