// LLM代理服务 - Web版 JavaScript (修复版)
// 修复按钮点击无响应的问题

class LLMProxyApp {
    constructor() {
        this.socket = null;
        this.currentConfig = null;
        this.isInitialized = false;
        this.init();
    }

    init() {
        try {
            console.log('初始化LLM代理服务应用...');
            this.initSocket();
            this.initEventListeners();
            this.loadConfig();
            this.updateWebUrl();
            this.isInitialized = true;
            console.log('应用初始化完成');
        } catch (error) {
            console.error('应用初始化失败:', error);
            this.showNotification('应用初始化失败: ' + error.message, 'error');
        }
    }

    initSocket() {
        try {
            // 检查Socket.IO是否可用
            if (typeof io === 'undefined') {
                console.warn('Socket.IO未定义，使用轮询模式');
                this.startPolling();
                return;
            }

            // 连接Socket.IO
            this.socket = io();
            
            this.socket.on('connect', () => {
                console.log('Socket连接成功');
                this.showNotification('连接成功', 'success');
            });

            this.socket.on('disconnect', () => {
                console.log('Socket连接断开');
                this.showNotification('连接断开', 'error');
            });

            this.socket.on('server_status', (data) => {
                console.log('收到服务器状态更新:', data);
                this.updateServerStatus(data);
            });

            this.socket.on('connect_error', (error) => {
                console.error('Socket连接错误:', error);
                this.showNotification('连接错误: ' + error.message, 'error');
                // 连接失败时启动轮询
                this.startPolling();
            });
        } catch (error) {
            console.error('Socket初始化失败:', error);
            this.startPolling();
        }
    }

    startPolling() {
        // 每5秒轮询一次服务器状态
        setInterval(() => {
            this.checkServerStatus();
        }, 5000);
    }

    async checkServerStatus() {
        try {
            const response = await fetch('/api/server/status');
            const data = await response.json();
            this.updateServerStatus({
                status: data.is_running ? 'running' : 'stopped',
                url: data.is_running ? 'http://localhost:8080' : null
            });
        } catch (error) {
            console.error('检查服务器状态失败:', error);
        }
    }

    initEventListeners() {
        try {
            // 等待DOM完全加载
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', () => {
                    this.bindEvents();
                });
            } else {
                this.bindEvents();
            }
        } catch (error) {
            console.error('事件监听器初始化失败:', error);
        }
    }

    bindEvents() {
        // 侧边栏导航
        const navLinks = document.querySelectorAll('.sidebar .nav-link');
        navLinks.forEach(link => {
            if (link) {
                link.addEventListener('click', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    const tabName = link.dataset.tab;
                    if (tabName) {
                        this.switchTab(tabName);
                    }
                });
            }
        });

        // 按钮事件 - 使用更健壮的选择器
        this.bindButtonEvent('save-config-btn', () => this.saveConfig());
        this.bindButtonEvent('reload-config-btn', () => this.loadConfig());
        this.bindButtonEvent('start-server-btn', () => this.startServer());
        this.bindButtonEvent('stop-server-btn', () => this.stopServer());

        // 实时配置更新
        this.setupRealTimeConfigUpdate();
    }

    bindButtonEvent(buttonId, handler) {
        const button = document.getElementById(buttonId);
        if (button) {
            // 移除之前的事件监听器（防止重复绑定）
            button.replaceWith(button.cloneNode(true));
            
            // 获取新的按钮引用
            const newButton = document.getElementById(buttonId);
            if (newButton) {
                newButton.addEventListener('click', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    
                    if (!newButton.disabled) {
                        handler();
                    }
                });
                
                // 添加键盘支持
                newButton.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        if (!newButton.disabled) {
                            handler();
                        }
                    }
                });
                
                console.log(`按钮 ${buttonId} 事件绑定成功`);
            }
        } else {
            console.error(`按钮 ${buttonId} 未找到`);
        }
    }

    setupRealTimeConfigUpdate() {
        const configInputs = [
            'api-port', 'api-host', 'web-port', 'web-host',
            'api-key', 'min-length', 'timeout', 'base-url',
            'group1-keys', 'group2-keys'
        ];

        configInputs.forEach(inputId => {
            const element = document.getElementById(inputId);
            if (element) {
                // 使用防抖保存配置
                let timeout;
                element.addEventListener('input', () => {
                    clearTimeout(timeout);
                    timeout = setTimeout(() => {
                        this.saveConfig();
                    }, 1000);
                });
            }
        });
    }

    async switchTab(tabName) {
        try {
            // 更新导航状态
            const navLinks = document.querySelectorAll('.sidebar .nav-link');
            navLinks.forEach(link => link.classList.remove('active'));
            
            const activeLink = document.querySelector(`[data-tab="${tabName}"]`);
            if (activeLink) {
                activeLink.classList.add('active');
            }

            // 切换内容
            const tabContents = document.querySelectorAll('.tab-content');
            tabContents.forEach(content => content.classList.remove('active'));
            
            const targetTab = document.getElementById(`${tabName}-tab`);
            if (targetTab) {
                targetTab.classList.add('active');
            }
        } catch (error) {
            console.error('切换标签页失败:', error);
        }
    }

    async loadConfig() {
        try {
            console.log('正在加载配置...');
            const response = await fetch('/api/config');
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            this.currentConfig = data;
            this.updateConfigUI(data);
            this.showNotification('配置加载成功', 'success');
            console.log('配置加载完成');
        } catch (error) {
            console.error('加载配置失败:', error);
            this.showNotification('加载配置失败: ' + error.message, 'error');
        }
    }

    updateConfigUI(data) {
        try {
            // 更新服务器配置
            if (data.server) {
                this.setInputValue('api-port', data.server.port);
                this.setInputValue('api-host', data.server.host);
                this.setInputValue('web-port', data.server.web_port);
                this.setInputValue('web-host', data.server.web_host);
                this.setInputValue('api-key', data.server.api_key);
                this.setInputValue('min-length', data.server.min_response_length);
                this.setInputValue('timeout', data.server.request_timeout);
            }

            // 更新基础URL
            if (data.base_url) {
                this.setInputValue('base-url', data.base_url);
            }

            // 更新API密钥
            if (data.api_keys) {
                this.setInputValue('group1-keys', data.api_keys.group1.join('\n'));
                this.setInputValue('group2-keys', data.api_keys.group2.join('\n'));
            }
        } catch (error) {
            console.error('更新UI失败:', error);
        }
    }

    setInputValue(elementId, value) {
        const element = document.getElementById(elementId);
        if (element) {
            element.value = value;
        }
    }

    async saveConfig() {
        try {
            console.log('正在保存配置...');
            const config = this.getConfigFromUI();
            
            const response = await fetch('/api/config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(config)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP ${response.status}`);
            }

            const result = await response.json();
            if (result.success) {
                this.showNotification('配置保存成功', 'success');
                this.currentConfig = config;
                console.log('配置保存完成');
            } else {
                throw new Error(result.error || '保存配置失败');
            }
        } catch (error) {
            console.error('保存配置失败:', error);
            this.showNotification('保存配置失败: ' + error.message, 'error');
        }
    }

    getConfigFromUI() {
        try {
            return {
                server: {
                    port: this.parseIntValue('api-port', 8080),
                    host: this.getInputValue('api-host', '0.0.0.0'),
                    web_port: this.parseIntValue('web-port', 5000),
                    web_host: this.getInputValue('web-host', '127.0.0.1'),
                    api_key: this.getInputValue('api-key', '123'),
                    min_response_length: this.parseIntValue('min-length', 400),
                    request_timeout: this.parseIntValue('timeout', 30)
                },
                api_keys: {
                    group1: this.getTextareaValues('group1-keys'),
                    group2: this.getTextareaValues('group2-keys')
                },
                base_url: this.getInputValue('base-url', 'https://generativelanguage.googleapis.com/v1beta')
            };
        } catch (error) {
            console.error('获取配置失败:', error);
            throw error;
        }
    }

    getInputValue(elementId, defaultValue = '') {
        const element = document.getElementById(elementId);
        return element ? element.value : defaultValue;
    }

    parseIntValue(elementId, defaultValue = 0) {
        const value = this.getInputValue(elementId, defaultValue.toString());
        const parsed = parseInt(value);
        return isNaN(parsed) ? defaultValue : parsed;
    }

    getTextareaValues(elementId) {
        const element = document.getElementById(elementId);
        if (!element) return [];
        
        return element.value
            .split('\n')
            .map(key => key.trim())
            .filter(key => key.length > 0);
    }

    async startServer() {
        try {
            console.log('正在启动服务器...');
            this.setButtonLoading('start-server-btn', true);
            
            const response = await fetch('/api/server/start', {
                method: 'POST'
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP ${response.status}`);
            }

            const result = await response.json();
            if (result.success) {
                this.showNotification('API服务器启动成功', 'success');
                this.updateServerButtons(true);
                console.log('服务器启动完成');
            } else {
                throw new Error(result.error || '启动服务器失败');
            }
        } catch (error) {
            console.error('启动服务器失败:', error);
            this.showNotification('启动服务器失败: ' + error.message, 'error');
        } finally {
            this.setButtonLoading('start-server-btn', false);
        }
    }

    async stopServer() {
        try {
            console.log('正在停止服务器...');
            this.setButtonLoading('stop-server-btn', true);
            
            const response = await fetch('/api/server/stop', {
                method: 'POST'
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP ${response.status}`);
            }

            const result = await response.json();
            if (result.success) {
                this.showNotification('API服务器停止成功', 'success');
                this.updateServerButtons(false);
                console.log('服务器停止完成');
            } else {
                throw new Error(result.error || '停止服务器失败');
            }
        } catch (error) {
            console.error('停止服务器失败:', error);
            this.showNotification('停止服务器失败: ' + error.message, 'error');
        } finally {
            this.setButtonLoading('stop-server-btn', false);
        }
    }

    updateServerStatus(data) {
        try {
            const statusIndicator = document.getElementById('status-indicator');
            const apiStatus = document.getElementById('api-status');
            const apiUrl = document.getElementById('api-url');

            if (!statusIndicator || !apiStatus || !apiUrl) {
                console.error('状态元素未找到');
                return;
            }

            if (data.status === 'running') {
                statusIndicator.className = 'badge bg-success';
                statusIndicator.innerHTML = '<i class="fas fa-circle"></i> 服务运行中';
                
                apiStatus.className = 'badge bg-success';
                apiStatus.textContent = '运行中';
                
                if (data.url) {
                    apiUrl.textContent = data.url;
                }
                
                this.updateServerButtons(true);
            } else {
                statusIndicator.className = 'badge bg-secondary';
                statusIndicator.innerHTML = '<i class="fas fa-circle"></i> 服务未运行';
                
                apiStatus.className = 'badge bg-secondary';
                apiStatus.textContent = '未运行';
                
                apiUrl.textContent = '-';
                
                this.updateServerButtons(false);
            }
        } catch (error) {
            console.error('更新服务器状态失败:', error);
        }
    }

    updateServerButtons(isRunning) {
        try {
            const startBtn = document.getElementById('start-server-btn');
            const stopBtn = document.getElementById('stop-server-btn');

            if (startBtn) {
                startBtn.disabled = isRunning;
            }
            if (stopBtn) {
                stopBtn.disabled = !isRunning;
            }
        } catch (error) {
            console.error('更新按钮状态失败:', error);
        }
    }

    updateWebUrl() {
        try {
            const webUrl = document.getElementById('web-url');
            if (webUrl) {
                const protocol = window.location.protocol;
                const host = window.location.host;
                webUrl.textContent = `${protocol}//${host}`;
            }
        } catch (error) {
            console.error('更新Web URL失败:', error);
        }
    }

    setButtonLoading(buttonId, isLoading) {
        try {
            const button = document.getElementById(buttonId);
            if (!button) {
                console.error(`按钮 ${buttonId} 未找到`);
                return;
            }

            if (isLoading) {
                button.disabled = true;
                const originalText = button.innerHTML;
                button.dataset.originalText = originalText;
                button.innerHTML = '<span class="loading"></span> 处理中...';
            } else {
                button.disabled = false;
                if (button.dataset.originalText) {
                    button.innerHTML = button.dataset.originalText;
                }
            }
        } catch (error) {
            console.error('设置按钮加载状态失败:', error);
        }
    }

    showNotification(message, type = 'info') {
        try {
            const container = document.getElementById('notification-container');
            if (!container) {
                console.warn('通知容器未找到:', message);
                return;
            }

            const notification = document.createElement('div');
            notification.className = `notification ${type}`;
            notification.textContent = message;

            container.appendChild(notification);

            // 显示通知
            setTimeout(() => {
                notification.classList.add('show');
            }, 100);

            // 自动隐藏
            setTimeout(() => {
                notification.classList.remove('show');
                setTimeout(() => {
                    if (notification.parentNode) {
                        notification.parentNode.removeChild(notification);
                    }
                }, 300);
            }, 3000);
        } catch (error) {
            console.error('显示通知失败:', error);
        }
    }
}

// 页面加载完成后初始化应用
document.addEventListener('DOMContentLoaded', () => {
    try {
        console.log('DOM已加载，初始化应用...');
        
        // 确保所有依赖项都可用
        if (typeof fetch === 'undefined') {
            console.error('Fetch API不可用');
            return;
        }
        
        window.llmProxyApp = new LLMProxyApp();
        console.log('应用初始化成功');
    } catch (error) {
        console.error('应用初始化失败:', error);
    }
});

// 页面卸载时清理资源
window.addEventListener('beforeunload', () => {
    try {
        if (window.llmProxyApp && window.llmProxyApp.socket) {
            window.llmProxyApp.socket.disconnect();
        }
    } catch (error) {
        console.error('清理资源失败:', error);
    }
});

// 添加全局错误处理
window.addEventListener('error', (event) => {
    console.error('全局错误:', {
        message: event.message,
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno,
        error: event.error
    });
});

window.addEventListener('unhandledrejection', (event) => {
    console.error('未处理的Promise拒绝:', event.reason);
});