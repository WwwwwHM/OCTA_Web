<template>
  <!-- ==================== 历史记录页面容器 ==================== -->
  <div class="history-container">
    <!-- 页面标题卡片 -->
    <el-card class="card-container">
      <template #header>
        <div class="card-header">
          <span>分割历史记录</span>
          <el-button 
            type="primary" 
            :loading="isRefreshing"
            @click="fetchHistory"
            style="margin-left: 10px;"
          >
            刷新
          </el-button>
        </div>
      </template>

      <!-- ==================== 统计信息卡片 ==================== -->
      <div class="statistics-container" v-if="statistics">
        <el-row :gutter="20" class="stat-row">
          <el-col :xs="12" :sm="8" :md="6">
            <div class="stat-card">
              <span class="stat-label">总分割数</span>
              <span class="stat-value">{{ statistics.total }}</span>
            </div>
          </el-col>
          <el-col :xs="12" :sm="8" :md="6">
            <div class="stat-card">
              <span class="stat-label">U-Net模型</span>
              <span class="stat-value">{{ statistics.unet }}</span>
            </div>
          </el-col>
          <el-col :xs="12" :sm="8" :md="6">
            <div class="stat-card">
              <span class="stat-label">FCN模型</span>
              <span class="stat-value">{{ statistics.fcn }}</span>
            </div>
          </el-col>
        </el-row>
      </div>

      <!-- ==================== 加载状态提示 ==================== -->
      <el-empty 
        v-if="!isLoading && historyList.length === 0" 
        description="暂无分割记录"
      />

      <!-- ==================== 分割记录表格 ==================== -->
      <el-table
        v-else
        :data="historyList"
        stripe
        highlight-current-row
        class="history-table"
        :loading="isLoading"
        style="width: 100%"
      >
        <!-- 序号列 -->
        <el-table-column type="index" label="序号" width="50" align="center" />

        <!-- 文件名列 -->
        <el-table-column 
          label="文件名" 
          prop="filename"
          min-width="180"
          show-overflow-tooltip
        >
          <template #default="{ row }">
            <span class="filename-text">{{ formatFilename(row.filename) }}</span>
          </template>
        </el-table-column>

        <!-- 上传时间列 -->
        <el-table-column 
          label="上传时间" 
          prop="upload_time"
          min-width="150"
          align="center"
        >
          <template #default="{ row }">
            <span>{{ formatTime(row.upload_time) }}</span>
          </template>
        </el-table-column>

        <!-- 模型类型列 -->
        <el-table-column 
          label="模型类型" 
          prop="model_type"
          width="100"
          align="center"
        >
          <template #default="{ row }">
            <!-- 使用标签显示模型类型，U-Net为蓝色，FCN为绿色 -->
            <el-tag 
              :type="row.model_type === 'unet' ? 'primary' : 'success'"
            >
              {{ row.model_type === 'unet' ? 'U-Net' : 'FCN' }}
            </el-tag>
          </template>
        </el-table-column>

        <!-- 操作列：预览、下载、删除 -->
        <el-table-column 
          label="操作" 
          width="280"
          align="center"
        >
          <template #default="{ row }">
            <!-- 预览原图按钮：使用el-button + el-dialog实现 -->
            <el-button 
              link 
              type="primary"
              @click="showImageDialog('original', row)"
            >
              <el-icon><View /></el-icon>
              原图预览
            </el-button>

            <!-- 预览分割结果按钮 -->
            <el-button 
              link 
              type="success"
              @click="showImageDialog('result', row)"
            >
              <el-icon><View /></el-icon>
              结果预览
            </el-button>

            <!-- 下载结果按钮 -->
            <el-button 
              link 
              type="info"
              @click="downloadImage(row)"
            >
              <el-icon><Download /></el-icon>
              下载
            </el-button>

            <!-- 删除按钮：弹窗确认后调用后端DELETE接口 -->
            <el-button 
              link 
              type="danger"
              @click="deleteRecord(row)"
            >
              <el-icon><Delete /></el-icon>
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- ==================== 图像预览对话框 ==================== -->
    <!-- 使用el-dialog包装el-image，支持全屏预览和放大缩小 -->
    <el-dialog 
      v-model="imageDialogVisible"
      :title="imageDialogTitle"
      width="80%"
      :close-on-click-modal="true"
      :show-close="true"
      @close="clearImageDialog"
    >
      <div class="image-preview-container">
        <!-- 使用el-image组件，支持预览、放大等功能 -->
        <el-image
          v-if="currentImageUrl"
          :src="currentImageUrl"
          fit="contain"
          class="preview-image"
          :preview-src-list="[currentImageUrl]"
          @error="handleImageLoadError"
        />
      </div>
    </el-dialog>
  </div>
</template>

<script setup>
/**
 * OCTA图像分割历史记录页面
 * 
 * 功能：
 * 1. 展示所有分割记录，包括文件名、上传时间、模型类型
 * 2. 支持预览原图和分割结果（el-image全屏预览）
 * 3. 支持下载分割结果
 * 4. 支持删除历史记录（调用后端DELETE接口）
 * 5. 显示统计信息（总数、各模型数量）
 * 6. 响应式布局，适配不同屏幕大小
 * 
 * API接口：
 * - GET /history/ : 获取所有分割记录
 * - DELETE /history/{id} : 删除指定记录（需后端实现）
 * - GET /images/{filename} : 获取原图
 * - GET /results/{filename} : 获取分割结果
 */

import { ref, computed, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { View, Download, Delete } from '@element-plus/icons-vue'
import axios from 'axios'

// ==================== 页面状态数据 ====================

// 历史记录列表
const historyList = ref([])

// 加载状态（正在从后端获取数据）
const isLoading = ref(false)

// 刷新状态（用户点击刷新按钮时的加载状态）
const isRefreshing = ref(false)

// 图像预览对话框的显示状态
const imageDialogVisible = ref(false)

// 图像预览对话框的标题（"原图预览"或"结果预览"）
const imageDialogTitle = ref('')

// 当前预览的图像URL
const currentImageUrl = ref('')

// ==================== 计算属性：统计信息 ====================

/**
 * 统计信息
 * 计算总记录数、U-Net模型数量、FCN模型数量
 */
const statistics = computed(() => {
  if (historyList.value.length === 0) return null
  
  const total = historyList.value.length
  const unet = historyList.value.filter(record => record.model_type === 'unet').length
  const fcn = historyList.value.filter(record => record.model_type === 'fcn').length
  
  return { total, unet, fcn }
})

// ==================== API基础地址 ====================

// 后端服务地址（需要与main.js中的axios baseURL保持一致）
const API_BASE_URL = 'http://127.0.0.1:8000'

// ==================== 核心函数：获取历史记录 ====================

/**
 * 从后端获取分割历史记录
 * 调用 GET /history/ 接口
 * 将返回的数组按upload_time倒序排列（后端已排序，此处备用）
 */
const fetchHistory = async () => {
  // 设置加载状态
  isRefreshing.value = true
  isLoading.value = true
  
  try {
    // 调用后端GET /history/接口
    const response = await axios.get(`${API_BASE_URL}/history/`)
    
    // 验证响应状态
    if (response.status === 200) {
      // 响应数据应该是一个数组
      historyList.value = response.data || []
      
      // 如果是刷新操作，显示成功提示
      if (isRefreshing.value) {
        ElMessage.success(`成功刷新！当前共 ${historyList.value.length} 条记录`)
      }
    } else {
      ElMessage.error('获取历史记录失败')
    }
  } catch (error) {
    // 异常处理：网络错误或后端错误
    console.error('获取历史记录异常:', error)
    
    if (error.response?.status === 500) {
      ElMessage.error('后端服务异常，请检查服务状态')
    } else if (error.code === 'ERR_NETWORK') {
      ElMessage.error('网络连接失败，请检查后端是否运行')
    } else {
      ElMessage.error('获取历史记录失败: ' + (error.message || '未知错误'))
    }
  } finally {
    // 关闭加载状态
    isLoading.value = false
    isRefreshing.value = false
  }
}

// ==================== 图像预览函数 ====================

/**
 * 显示图像预览对话框
 * 
 * 参数：
 * - type: 'original' 预览原图，'result' 预览分割结果
 * - row: 表格行数据，包含filename和result_path等信息
 */
const showImageDialog = (type, row) => {
  // 确定预览的图像URL
  if (type === 'original') {
    // 预览原图：使用 /images/{filename} 接口
    currentImageUrl.value = `${API_BASE_URL}/images/${row.filename}`
    imageDialogTitle.value = '原图预览'
  } else if (type === 'result') {
    // 预览分割结果：从result_path提取文件名，使用 /results/{filename} 接口
    const resultFilename = row.result_path.split('/').pop()
    currentImageUrl.value = `${API_BASE_URL}/results/${resultFilename}`
    imageDialogTitle.value = '分割结果预览'
  }
  
  // 打开预览对话框
  imageDialogVisible.value = true
}

/**
 * 清空图像预览对话框
 * 在对话框关闭时调用，清除URL和标题
 */
const clearImageDialog = () => {
  currentImageUrl.value = ''
  imageDialogTitle.value = ''
}

/**
 * 处理图像加载错误
 * 如果预览图像加载失败，显示错误提示
 */
const handleImageLoadError = () => {
  ElMessage.error('图像加载失败，请检查文件是否存在')
}

// ==================== 下载函数 ====================

/**
 * 下载分割结果
 * 
 * 参数：
 * - row: 表格行数据
 * 
 * 流程：
 * 1. 从result_path提取结果文件名
 * 2. 构建下载URL: /results/{filename}
 * 3. 创建临时<a>元素，触发浏览器下载
 * 4. 清理临时元素
 */
const downloadImage = (row) => {
  try {
    // 从路径提取结果文件名（如 ./results/xxx_segmented.png -> xxx_segmented.png）
    const resultFilename = row.result_path.split('/').pop()
    
    // 构建完整的下载URL
    const downloadUrl = `${API_BASE_URL}/results/${resultFilename}`
    
    // 创建临时<a>元素
    const link = document.createElement('a')
    link.href = downloadUrl
    
    // 设置下载文件名（显示在浏览器下载对话框中）
    // 使用原始文件名_segmented.png作为下载文件名
    link.download = resultFilename
    
    // 将元素添加到DOM中（某些浏览器需要）
    document.body.appendChild(link)
    
    // 触发点击事件，开始下载
    link.click()
    
    // 清理：移除临时元素
    document.body.removeChild(link)
    
    // 显示成功提示
    ElMessage.success('开始下载分割结果')
  } catch (error) {
    // 错误处理
    console.error('下载失败:', error)
    ElMessage.error('下载失败: ' + (error.message || '未知错误'))
  }
}

// ==================== 删除函数 ====================

/**
 * 删除分割记录
 * 
 * 参数：
 * - row: 表格行数据，包含id字段
 * 
 * 流程：
 * 1. 弹窗确认删除
 * 2. 用户确认后，调用后端 DELETE /history/{id} 接口
 * 3. 删除成功后，从前端列表中移除该记录
 * 4. 显示成功/失败提示
 * 
 * 注意：需要后端实现DELETE /history/{id}接口
 */
const deleteRecord = (row) => {
  // 弹窗确认删除
  ElMessageBox.confirm(
    `确定要删除这条分割记录吗？\n文件名: ${row.filename}`,
    '删除确认',
    {
      confirmButtonText: '确定删除',
      cancelButtonText: '取消',
      type: 'warning',
    }
  ).then(async () => {
    // 用户点击确认按钮
    try {
      // 调用后端DELETE接口删除记录
      // 注意：需要后端实现此接口
      const response = await axios.delete(`${API_BASE_URL}/history/${row.id}`)
      
      if (response.status === 200) {
        // 删除成功，从前端列表中移除
        const index = historyList.value.findIndex(item => item.id === row.id)
        if (index !== -1) {
          historyList.value.splice(index, 1)
        }
        ElMessage.success('记录已删除')
      }
    } catch (error) {
      // 异常处理
      console.error('删除记录失败:', error)
      
      // 特殊处理：如果是404说明后端没有实现DELETE接口
      if (error.response?.status === 404) {
        ElMessage.error('后端DELETE接口未实现，请联系开发人员')
      } else if (error.response?.status === 500) {
        ElMessage.error('删除失败: 后端服务异常')
      } else {
        ElMessage.error('删除失败: ' + (error.message || '未知错误'))
      }
    }
  }).catch(() => {
    // 用户点击取消按钮或关闭对话框
    // 不进行任何操作
  })
}

// ==================== 格式化函数 ====================

/**
 * 格式化文件名
 * 将UUID格式的文件名显示为简短形式
 * 例如：a1b2c3d4-e5f6-7890-abcd-ef1234567890.png -> a1b2c3d4...png
 */
const formatFilename = (filename) => {
  if (!filename || filename.length <= 20) {
    return filename
  }
  // 只显示前8位和扩展名
  return filename.substring(0, 8) + '...' + filename.substring(filename.lastIndexOf('.'))
}

/**
 * 格式化时间
 * 将数据库中的时间字符串转换为易读的格式
 * 输入格式：2026-01-12 14:30:25
 * 输出格式：01-12 14:30 （省略年份和秒钟，提高表格紧凑性）
 */
const formatTime = (timeString) => {
  if (!timeString) return '-'
  // 提取时间中的月-日 时:分
  const parts = timeString.split(' ')
  if (parts.length >= 2) {
    const datePart = parts[0].substring(5) // 取年份后的部分：01-12
    const timePart = parts[1].substring(0, 5) // 取时:分部分：14:30
    return `${datePart} ${timePart}`
  }
  return timeString
}

// ==================== 生命周期：页面挂载时获取数据 ====================

/**
 * 页面首次加载时，自动从后端获取历史记录
 * 这样用户打开历史记录页面时，能立即看到数据
 */
onMounted(() => {
  fetchHistory()
})
</script>

<style scoped>
/**
 * ==================== 页面容器样式 ====================
 * 整体布局和主容器的样式
 */

.history-container {
  /* 容器最大宽度限制，超出此宽度时保持居中 */
  max-width: 1400px;
  margin: 20px auto;
  padding: 0 10px;
  /* 动画过渡效果，提升用户体验 */
  animation: slideIn 0.3s ease-in-out;
}

/* 页面加载动画：从上方滑入 */
@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateY(-20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/**
 * ==================== 卡片容器样式 ====================
 * Element Plus el-card组件的自定义样式
 */

.card-container {
  /* 蓝色系统主题，与HomeView保持一致 */
  background: linear-gradient(135deg, #f5f7fa 0%, #ffffff 100%);
  border: 1px solid #e0e6f2;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.06);
  /* 悬停时增加阴影，提升交互反馈 */
  transition: all 0.3s ease;
}

.card-container:hover {
  box-shadow: 0 4px 16px 0 rgba(0, 0, 0, 0.1);
}

/* 卡片头部样式 */
.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 18px;
  font-weight: 600;
  color: #409eff;
}

/**
 * ==================== 统计信息容器 ====================
 * 显示总数、各模型数量的小卡片
 */

.statistics-container {
  margin-bottom: 20px;
  padding: 10px 0;
  border-bottom: 1px solid #e0e6f2;
}

.stat-row {
  margin-bottom: 0;
}

/* 单个统计卡片 */
.stat-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 15px;
  background: linear-gradient(135deg, #e0edff 0%, #f0f9ff 100%);
  border-radius: 6px;
  border-left: 4px solid #409eff;
  /* 响应式：在移动设备上减少内边距 */
  transition: all 0.3s ease;
}

.stat-card:hover {
  background: linear-gradient(135deg, #c6e2ff 0%, #e0edff 100%);
  transform: translateY(-2px);
}

.stat-label {
  font-size: 12px;
  color: #606266;
  margin-bottom: 8px;
}

.stat-value {
  font-size: 28px;
  font-weight: bold;
  color: #409eff;
}

/**
 * ==================== 表格样式 ====================
 * 历史记录表格的样式
 */

.history-table {
  margin-top: 20px;
}

/* 表格头部样式 */
:deep(.el-table__header) {
  background-color: #f5f7fa;
}

:deep(.el-table__header th) {
  background-color: #f5f7fa;
  color: #606266;
  font-weight: 600;
}

/* 表格行样式 */
:deep(.el-table__body tr:hover > td) {
  background-color: #f0f9ff !important;
}

/* 文件名文本样式 */
.filename-text {
  color: #303133;
  font-family: 'Monaco', 'Menlo', monospace;
  font-size: 12px;
}

/**
 * ==================== 操作按钮组样式 ====================
 * 表格操作列中的按钮
 */

:deep(.el-button--text) {
  /* 链接样式按钮，占用空间小 */
  padding: 0 4px;
  margin: 0 2px;
}

:deep(.el-button--text.is-disabled) {
  color: #c0c4cc;
  cursor: not-allowed;
}

/**
 * ==================== 图像预览对话框样式 ====================
 * el-dialog和图像容器的样式
 */

.image-preview-container {
  display: flex;
  align-items: center;
  justify-content: center;
  /* 最小高度确保有足够的空间显示图像 */
  min-height: 300px;
  background-color: #fafbfc;
  border-radius: 6px;
  overflow: auto;
}

/* el-image组件的样式 */
.preview-image {
  max-width: 100%;
  max-height: 70vh;
  border-radius: 4px;
}

/**
 * ==================== 响应式设计 ====================
 * 根据不同屏幕大小调整布局
 */

/* 超小屏幕：手机竖屏 (< 576px) */
@media (max-width: 576px) {
  .history-container {
    padding: 0 5px;
    margin: 10px auto;
  }
  
  .card-container {
    border-radius: 4px;
  }
  
  .card-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 10px;
  }
  
  .stat-card {
    padding: 10px;
  }
  
  .stat-value {
    font-size: 20px;
  }
  
  :deep(.el-table__body) {
    font-size: 12px;
  }
}

/* 小屏幕：手机横屏或平板 (576px - 992px) */
@media (max-width: 992px) {
  .history-container {
    max-width: 100%;
  }
  
  /* 在小屏幕上隐藏部分信息 */
  :deep(.el-table__body) {
    font-size: 13px;
  }
}

/* 大屏幕：桌面设备 (> 992px) */
@media (min-width: 992px) {
  .history-container {
    max-width: 1400px;
  }
}

/**
 * ==================== 元素Plus主题覆盖 ====================
 * 覆盖默认的Element Plus样式，保持设计一致性
 */

/* 空状态样式 */
:deep(.el-empty) {
  padding: 40px 20px;
}

:deep(.el-empty__image) {
  height: 160px;
}

/* 消息框样式 */
:deep(.el-message-box) {
  border-radius: 8px;
}

/* 按钮样式 */
:deep(.el-button) {
  border-radius: 4px;
  transition: all 0.3s ease;
}

:deep(.el-button--primary) {
  background: linear-gradient(90deg, #409eff 0%, #66b1ff 100%);
  border: none;
}

:deep(.el-button--primary:hover) {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(64, 158, 255, 0.4);
}

/* 标签样式 */
:deep(.el-tag) {
  border-radius: 4px;
  padding: 4px 8px;
}
</style>
