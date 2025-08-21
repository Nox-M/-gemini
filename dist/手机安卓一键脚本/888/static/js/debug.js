// 增强调试脚本 - 检查按钮点击问题
console.log('=== LLM代理服务调试信息 ===');

// 检查页面加载状态
document.addEventListener('DOMContentLoaded', function() {
    console.log('✅ DOM已加载');
    
    // 检查所有关键元素
    const elements = {
        'start-server-btn': document.getElementById('start-server-btn'),
        'stop-server-btn': document.getElementById('stop-server-btn'),
        'save-config-btn': document.getElementById('save-config-btn'),
        'reload-config-btn': document.getElementById('reload-config-btn'),
        'notification-container': document.getElementById('notification-container')
    };
    
    console.log('=== 元素检查 ===');
    Object.keys(elements).forEach(id => {
        const el = elements[id];
        if (el) {
            console.log(`✅ ${id}: 存在`);
            console.log(`   - 禁用状态: ${el.disabled}`);
            console.log(`   - 可见性: ${window.getComputedStyle(el).display}`);
            console.log(`   - 指针事件: ${window.getComputedStyle(el).pointerEvents}`);
            console.log(`   - z-index: ${window.getComputedStyle(el).zIndex}`);
            console.log(`   - 位置: ${window.getComputedStyle(el).position}`);
        } else {
            console.error(`❌ ${id}: 不存在`);
        }
    });
    
    // 检查Socket.IO连接
    console.log('=== Socket.IO检查 ===');
    if (typeof io !== 'undefined') {
        console.log('✅ Socket.IO库已加载');
        
        // 创建测试连接
        const testSocket = io();
        
        testSocket.on('connect', () => {
            console.log('✅ Socket.IO连接成功');
            testSocket.disconnect();
        });
        
        testSocket.on('connect_error', (error) => {
            console.error('❌ Socket.IO连接失败:', error);
        });
    } else {
        console.error('❌ Socket.IO库未加载');
    }
    
    // 检查事件监听器
    console.log('=== 事件监听器检查 ===');
    setTimeout(() => {
        // 手动测试按钮点击
        Object.keys(elements).forEach(id => {
            const el = elements[id];
            if (el && el.tagName === 'BUTTON') {
                el.addEventListener('click', (e) => {
                    console.log(`🖱️ 按钮 ${id} 被点击`, {
                        target: e.target,
                        disabled: e.target.disabled,
                        timestamp: new Date().toISOString()
                    });
                });
                
                // 添加悬停效果测试
                el.addEventListener('mouseenter', () => {
                    console.log(`🖱️ 鼠标悬停在 ${id} 上`);
                });
            }
        });
    }, 1000);
    
    // 检查是否有重叠元素
    console.log('=== 重叠元素检查 ===');
    setTimeout(() => {
        const buttons = document.querySelectorAll('button');
        buttons.forEach(btn => {
            const rect = btn.getBoundingClientRect();
            const centerX = rect.left + rect.width / 2;
            const centerY = rect.top + rect.height / 2;
            
            const elementsAtPoint = document.elementsFromPoint(centerX, centerY);
            if (elementsAtPoint.length > 1) {
                console.log(`⚠️ 按钮 ${btn.id || btn.textContent} 可能被其他元素覆盖:`, 
                           elementsAtPoint.map(el => el.tagName + (el.id ? `#${el.id}` : '')));
            }
        });
    }, 2000);
    
    // 检查JavaScript错误
    console.log('=== JavaScript错误检查 ===');
    window.addEventListener('error', (event) => {
        console.error('❌ JavaScript错误:', {
            message: event.message,
            filename: event.filename,
            lineno: event.lineno,
            colno: event.colno,
            error: event.error
        });
    });
    
    // 检查未捕获的Promise错误
    window.addEventListener('unhandledrejection', (event) => {
        console.error('❌ 未处理的Promise拒绝:', event.reason);
    });
    
    // 测试fetch API
    console.log('=== Fetch API测试 ===');
    fetch('/api/config')
        .then(response => {
            console.log('✅ /api/config 响应:', response.status);
            return response.json();
        })
        .then(data => console.log('✅ 配置数据:', data))
        .catch(error => console.error('❌ /api/config 错误:', error));
});

// 添加全局错误处理
window.addEventListener('load', () => {
    console.log('✅ 页面完全加载');
    
    // 检查是否有任何CSS阻止点击
    const allButtons = document.querySelectorAll('button');
    allButtons.forEach(btn => {
        const computedStyle = window.getComputedStyle(btn);
        if (computedStyle.pointerEvents === 'none') {
            console.error(`❌ 按钮 ${btn.id || btn.textContent} 的 pointer-events 被禁用`);
        }
        if (computedStyle.display === 'none') {
            console.error(`❌ 按钮 ${btn.id || btn.textContent} 被隐藏 (display: none)`);
        }
        if (computedStyle.visibility === 'hidden') {
            console.error(`❌ 按钮 ${btn.id || btn.textContent} 被隐藏 (visibility: hidden)`);
        }
    });
});

// 添加调试工具函数
window.debugTools = {
    testButtonClick: (buttonId) => {
        const btn = document.getElementById(buttonId);
        if (btn) {
            console.log(`🧪 测试点击 ${buttonId}`);
            btn.click();
        } else {
            console.error(`❌ 按钮 ${buttonId} 不存在`);
        }
    },
    
    checkButtonState: (buttonId) => {
        const btn = document.getElementById(buttonId);
        if (btn) {
            console.log(`📊 按钮 ${buttonId} 状态:`, {
                disabled: btn.disabled,
                display: window.getComputedStyle(btn).display,
                visibility: window.getComputedStyle(btn).visibility,
                pointerEvents: window.getComputedStyle(btn).pointerEvents,
                opacity: window.getComputedStyle(btn).opacity
            });
        }
    },
    
    listAllButtons: () => {
        const buttons = document.querySelectorAll('button');
        console.log('📋 所有按钮:');
        buttons.forEach((btn, index) => {
            console.log(`${index + 1}. ${btn.id || '无ID'} - ${btn.textContent.trim()}`);
        });
    }
};

console.log('🔧 调试工具已加载，可用命令:');
console.log('  debugTools.testButtonClick("start-server-btn")');
console.log('  debugTools.checkButtonState("save-config-btn")');
console.log('  debugTools.listAllButtons()');