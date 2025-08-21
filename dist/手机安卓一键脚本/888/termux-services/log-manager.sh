#!/data/data/com.termux/files/usr/bin/bash
# LLM代理服务 - 日志管理脚本
# 提供日志查看、清理和轮转功能

# 设置变量
LOG_DIR="$HOME/.llm-proxy/logs"
LOG_FILE="$LOG_DIR/llm-proxy.log"
ERROR_LOG="$LOG_DIR/llm-proxy-error.log"
MAX_LOG_SIZE=10485760  # 10MB
MAX_LOG_FILES=5

# 创建日志目录
mkdir -p "$LOG_DIR"

# 函数：检查日志大小并轮转
rotate_logs() {
    for logfile in "$LOG_FILE" "$ERROR_LOG"; do
        if [[ -f "$logfile" ]]; then
            local size=$(stat -c%s "$logfile" 2>/dev/null || echo 0)
            if [[ $size -gt $MAX_LOG_SIZE ]]; then
                echo "$(date): 日志文件 $logfile 超过最大大小，开始轮转..."
                
                # 删除最旧的日志文件
                if [[ -f "${logfile}.${MAX_LOG_FILES}" ]]; then
                    rm -f "${logfile}.${MAX_LOG_FILES}"
                fi
                
                # 轮转日志文件
                for i in $(seq $((MAX_LOG_FILES - 1)) -1 1); do
                    if [[ -f "${logfile}.${i}" ]]; then
                        mv "${logfile}.${i}" "${logfile}.$((i + 1))"
                    fi
                done
                
                # 重命名当前日志文件
                mv "$logfile" "${logfile}.1"
                
                # 创建新的空日志文件
                touch "$logfile"
                
                echo "$(date): 日志轮转完成"
            fi
        fi
    done
}

# 函数：查看日志
view_logs() {
    local lines=${1:-50}
    
    echo "=== LLM代理服务日志 (最近 $lines 行) ==="
    if [[ -f "$LOG_FILE" ]]; then
        tail -n "$lines" "$LOG_FILE"
    else
        echo "日志文件不存在: $LOG_FILE"
    fi
    
    echo -e "\n=== 错误日志 (最近 $lines 行) ==="
    if [[ -f "$ERROR_LOG" ]]; then
        tail -n "$lines" "$ERROR_LOG"
    else
        echo "错误日志文件不存在: $ERROR_LOG"
    fi
}

# 函数：清理日志
clean_logs() {
    echo "正在清理日志文件..."
    
    # 清理主日志
    if [[ -f "$LOG_FILE" ]]; then
        > "$LOG_FILE"
        echo "已清空主日志文件"
    fi
    
    # 清理错误日志
    if [[ -f "$ERROR_LOG" ]]; then
        > "$ERROR_LOG"
        echo "已清空错误日志文件"
    fi
    
    # 清理轮转的旧日志
    for logfile in "$LOG_FILE" "$ERROR_LOG"; do
        for i in $(seq 1 $MAX_LOG_FILES); do
            if [[ -f "${logfile}.${i}" ]]; then
                rm -f "${logfile}.${i}"
                echo "已删除旧日志文件: ${logfile}.${i}"
            fi
        done
    done
    
    echo "日志清理完成"
}

# 函数：查看服务状态
status() {
    local pid_file="$LOG_DIR/llm-proxy.pid"
    
    if [[ -f "$pid_file" ]]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo "服务状态: 运行中 (PID: $pid)"
            
            # 检查端口占用
            local port=$(grep -E "port.*=" config.ini | head -1 | sed 's/.*= *//')
            if [[ -n "$port" ]]; then
                if netstat -tuln 2>/dev/null | grep -q ":$port "; then
                    echo "端口状态: $port 端口正在监听"
                else
                    echo "端口状态: $port 端口未监听"
                fi
            fi
        else
            echo "服务状态: 未运行 (PID文件存在但进程不存在)"
        fi
    else
        echo "服务状态: 未运行 (PID文件不存在)"
    fi
}

# 函数：显示帮助信息
show_help() {
    echo "LLM代理服务 - 日志管理工具"
    echo
    echo "用法: $0 [命令]"
    echo
    echo "命令:"
    echo "  view [行数]    查看日志 (默认50行)"
    echo "  rotate         手动轮转日志"
    echo "  clean          清理所有日志文件"
    echo "  status         查看服务状态"
    echo "  help           显示此帮助信息"
    echo
}

# 主程序
case "${1:-help}" in
    "view")
        view_logs "${2:-50}"
        ;;
    "rotate")
        rotate_logs
        ;;
    "clean")
        clean_logs
        ;;
    "status")
        status
        ;;
    "help"|*)
        show_help
        ;;
esac