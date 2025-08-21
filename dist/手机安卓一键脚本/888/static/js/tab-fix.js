// 修复标签页切换问题的独立脚本
(function() {
    'use strict';
    
    function initTabSwitching() {
        console.log('初始化标签页切换...');
        
        // 获取所有导航按钮
        const navButtons = document.querySelectorAll('.nav-btn[data-tab]');
        console.log('找到导航按钮:', navButtons.length);
        
        // 为每个按钮添加点击事件
        navButtons.forEach(button => {
            // 确保只绑定一次事件
            button.removeEventListener('click', handleTabClick);
            button.addEventListener('click', handleTabClick);
        });
        
        function handleTabClick(event) {
            event.preventDefault();
            event.stopPropagation();
            
            const tabName = this.dataset.tab;
            console.log('切换到标签:', tabName);
            
            if (!tabName) return;
            
            switchTab(tabName);
        }
        
        function switchTab(tabName) {
            try {
                // 更新导航按钮状态
                const navButtons = document.querySelectorAll('.nav-btn');
                navButtons.forEach(btn => {
                    btn.classList.remove('active');
                });
                
                const activeButton = document.querySelector(`[data-tab="${tabName}"]`);
                if (activeButton) {
                    activeButton.classList.add('active');
                }
                
                // 切换内容区域
                const tabContents = document.querySelectorAll('.tab-content');
                tabContents.forEach(content => {
                    content.classList.remove('active');
                });
                
                const targetTab = document.getElementById(`${tabName}-tab`);
                if (targetTab) {
                    targetTab.classList.add('active');
                    console.log('成功切换到标签:', tabName);
                } else {
                    console.error('未找到标签页:', `${tabName}-tab`);
                }
                
                // 滚动到顶部
                window.scrollTo({ top: 0, behavior: 'smooth' });
                
            } catch (error) {
                console.error('切换标签页失败:', error);
            }
        }
        
        // 确保默认标签页可见
        const defaultTab = document.getElementById('quick-start-tab');
        if (defaultTab) {
            defaultTab.classList.add('active');
        }
    }
    
    // 在DOM加载完成后初始化
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initTabSwitching);
    } else {
        initTabSwitching();
    }
    
    // 确保在页面完全加载后也执行一次
    window.addEventListener('load', initTabSwitching);
    
    console.log('标签页切换修复脚本已加载');
})();