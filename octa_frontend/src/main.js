import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
// 引入Element Plus
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'

// ==================== 全局错误处理：忽略浏览器扩展消息通道错误 ====================
// 这个错误通常由浏览器扩展（如Vue DevTools）引起，不影响应用功能
window.addEventListener('error', (event) => {
  if (event.message && event.message.includes('message channel closed')) {
    event.preventDefault()
    console.warn('[App] 已忽略浏览器扩展消息通道错误（通常来自开发者工具扩展）')
    return false
  }
})

window.addEventListener('unhandledrejection', (event) => {
  if (event.reason && event.reason.message && event.reason.message.includes('message channel closed')) {
    event.preventDefault()
    console.warn('[App] 已忽略浏览器扩展Promise拒绝错误（通常来自开发者工具扩展）')
    return false
  }
})

// 创建App并挂载Element Plus
const app = createApp(App)
app.use(router)
app.use(ElementPlus) // 全局注册Element Plus
app.mount('#app')