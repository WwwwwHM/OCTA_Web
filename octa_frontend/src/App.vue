<script setup>
import { RouterView, useRoute } from 'vue-router'
import { computed } from 'vue'
import { HomeFilled, Clock, Guide, Folder, Upload, WarningFilled } from '@element-plus/icons-vue'
import { useGlobalState } from '@/composables/useGlobalState'

const route = useRoute()
const activePath = computed(() => route.path)

// 全局状态管理
const { rsUnet3PlusAvailable } = useGlobalState()
</script>

<template>
  <div class="layout">
    <header class="header">
      <div class="brand">
        <img alt="Vue logo" class="logo" src="@/assets/logo.svg" width="48" height="48" />
        <div class="title">OCTA 图像分割平台</div>
      </div>

      <el-menu
        class="nav-menu"
        mode="horizontal"
        router
        :default-active="activePath"
        background-color="#ffffff"
        text-color="#303133"
        active-text-color="#409EFF"
      >
        <el-menu-item index="/">
          <el-icon><HomeFilled /></el-icon>
          <span>首页</span>
        </el-menu-item>
        <el-menu-item index="/history">
          <el-icon><Clock /></el-icon>
          <span>历史记录</span>
        </el-menu-item>
        
        <el-menu-item index="/weight-manager">
          <el-icon><Upload /></el-icon>
          <span>权重管理</span>
        </el-menu-item>
        
        <el-menu-item index="/file-manager">
          <el-icon><Folder /></el-icon>
          <span>文件管理</span>
        </el-menu-item>
        <el-menu-item index="/about">
          <el-icon><Guide /></el-icon>
          <span>关于</span>
        </el-menu-item>
      </el-menu>
    </header>

    <main class="content">
      <RouterView />
    </main>
  </div>
</template>

<style scoped>
.layout {
  min-height: 100vh;
  background: #f5f7fa;
}

.header {
  position: sticky;
  top: 0;
  z-index: 10;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 24px;
  background: #ffffff;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}

.brand {
  display: flex;
  align-items: center;
  gap: 12px;
}

.title {
  font-size: 16px;
  font-weight: 600;
  color: #303133;
}

.logo {
  border-radius: 8px;
}

.nav-menu {
  border-bottom: none;
}

/* 子菜单样式优化 */
.el-sub-menu__title {
  font-weight: 500;
}

.disabled-hint {
  margin-left: 8px;
  color: #E6A23C;
  font-size: 14px;
}

/* 子菜单项增强 */
:deep(.el-menu--horizontal .el-sub-menu .el-menu-item) {
  padding-left: 40px;
  min-width: 220px;
}

:deep(.el-menu--horizontal .el-sub-menu .el-menu-item:hover:not(.is-disabled)) {
  background-color: #ecf5ff;
  color: #409EFF;
}

:deep(.el-menu--horizontal .el-sub-menu .el-menu-item.is-disabled) {
  opacity: 0.5;
  cursor: not-allowed;
}

.content {
  padding: 24px;
  max-width: 1280px;
  margin: 0 auto;
}

@media (max-width: 768px) {
  .header {
    flex-direction: column;
    align-items: flex-start;
    gap: 12px;
  }

  .nav-menu {
    width: 100%;
  }

  .content {
    padding: 16px;
  }
}
</style>
