<template>
  <div class="file-manager-container">
    <!-- 页面标题卡片 -->
    <el-card class="card-container header-card">
      <template #header>
        <div class="card-header">
          <span class="title-text">文件管理中心</span>
          <span class="subtitle-text">图片和数据集文件统一管理与复用</span>
        </div>
      </template>
    </el-card>

    <!-- 筛选和操作区域 -->
    <el-card class="card-container filter-card">
      <div class="filter-section">
        <div class="filter-left">
          <el-select
            v-model="filterType"
            placeholder="选择文件类型"
            style="width: 200px"
            @change="handleFilterChange"
          >
            <el-option label="全部文件" value="all"></el-option>
            <el-option label="图片" value="image"></el-option>
            <el-option label="数据集" value="dataset"></el-option>
            <el-option label="模型权重" value="weight"></el-option>
          </el-select>
          <el-button type="primary" :icon="Refresh" @click="fetchFileList" :loading="loading">
            刷新列表
          </el-button>
        </div>
        <div class="filter-right">
          <span class="file-count">共 {{ fileList.length }} 个文件</span>
        </div>
      </div>
    </el-card>

    <!-- 文件列表区域 -->
    <el-card class="card-container table-card" v-loading="loading" element-loading-text="加载中...">
      <el-table
        :data="fileList"
        stripe
        style="width: 100%"
        :header-cell-style="{ background: '#f5f7fa', color: '#606266' }"
      >
        <el-table-column prop="id" label="ID" width="80" align="center" />
        
        <el-table-column prop="file_name" label="文件名" min-width="200" show-overflow-tooltip>
          <template #default="scope">
            <span class="file-name">{{ scope.row.file_name }}</span>
          </template>
        </el-table-column>
        
        <el-table-column prop="file_type" label="类型" width="100" align="center">
          <template #default="scope">
            <el-tag
              :type="scope.row.file_type === 'image' ? 'primary' : scope.row.file_type === 'dataset' ? 'success' : 'warning'"
              size="small"
            >
              {{ 
                scope.row.file_type === 'image' ? '图片' : 
                scope.row.file_type === 'dataset' ? '数据集' : 
                '模型权重'
              }}
            </el-tag>
          </template>
        </el-table-column>
        
        <el-table-column prop="file_size" label="文件大小" width="120" align="center">
          <template #default="scope">
            {{ formatFileSize(scope.row.file_size) }}
          </template>
        </el-table-column>
        
        <el-table-column prop="upload_time" label="上传时间" width="180" align="center">
          <template #default="scope">
            {{ formatTime(scope.row.upload_time) }}
          </template>
        </el-table-column>
        
        <el-table-column prop="related_model" label="关联模型" min-width="200" show-overflow-tooltip>
          <template #default="scope">
            <span v-if="scope.row.related_model" class="model-path">
              {{ scope.row.related_model }}
            </span>
            <span v-else class="no-model">未关联</span>
          </template>
        </el-table-column>
        
        <el-table-column label="操作" width="240" align="center" fixed="right">
          <template #default="scope">
            <!-- 图片类型：测试分割 + 删除 -->
            <div v-if="scope.row.file_type === 'image'" class="action-buttons">
              <el-button
                type="warning"
                size="small"
                :icon="View"
                @click="handleTest(scope.row)"
                :loading="testingFileId === scope.row.id"
              >
                测试分割
              </el-button>
              <el-button
                type="danger"
                size="small"
                :icon="Delete"
                @click="handleDelete(scope.row)"
                :loading="deletingFileId === scope.row.id"
              >
                删除
              </el-button>
            </div>
            
            <!-- 数据集类型：重新训练 + 删除 -->
            <div v-else-if="scope.row.file_type === 'dataset'" class="action-buttons">
              <el-button
                type="primary"
                size="small"
                :icon="Promotion"
                @click="handleReuse(scope.row)"
                :loading="trainingFileId === scope.row.id"
              >
                重新训练
              </el-button>
              <el-button
                type="danger"
                size="small"
                :icon="Delete"
                @click="handleDelete(scope.row)"
                :loading="deletingFileId === scope.row.id"
              >
                删除
              </el-button>
            </div>
            
            <!-- 模型权重类型：仅删除 -->
            <div v-else class="action-buttons">
              <el-button
                type="danger"
                size="small"
                :icon="Delete"
                @click="handleDelete(scope.row)"
                :loading="deletingFileId === scope.row.id"
              >
                删除
              </el-button>
            </div>
          </template>
        </el-table-column>
      </el-table>

      <!-- 空状态提示 -->
      <el-empty v-if="!loading && fileList.length === 0" description="暂无文件数据" />
    </el-card>

    <!-- 重新训练弹窗 -->
    <el-dialog
      v-model="trainDialogVisible"
      title="重新训练模型"
      width="500px"
      :close-on-click-modal="false"
      @close="resetTrainForm"
    >
      <el-form ref="trainFormRef" :model="trainForm" label-width="100px">
        <el-form-item label="数据集名称">
          <el-input v-model="currentFile.file_name" disabled />
        </el-form-item>
        
        <el-form-item label="训练轮数">
          <el-input-number
            v-model="trainForm.epochs"
            :min="1"
            :max="100"
            :step="1"
            controls-position="right"
            style="width: 100%"
          />
          <div class="form-tip">建议值：10-30轮</div>
        </el-form-item>
        
        <el-form-item label="学习率">
          <el-input-number
            v-model="trainForm.lr"
            :min="0.00001"
            :max="0.01"
            :step="0.0001"
            :precision="5"
            controls-position="right"
            style="width: 100%"
          />
          <div class="form-tip">建议值：0.0001-0.001</div>
        </el-form-item>
      </el-form>
      
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="trainDialogVisible = false">取消</el-button>
          <el-button
            type="primary"
            @click="confirmTrain"
            :loading="isTraining"
          >
            {{ isTraining ? '训练中...' : '开始训练' }}
          </el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Refresh, Delete, View, Promotion } from '@element-plus/icons-vue'
import axios from 'axios'

// ==================== Vue Router ====================
const router = useRouter()

// ==================== 后端API配置 ====================
const API_BASE_URL = 'http://127.0.0.1:8000'

// ==================== 数据状态 ====================
// 筛选条件
const filterType = ref('all')

// 文件列表
const fileList = ref([])

// 加载状态
const loading = ref(false)

// 当前操作的文件
const currentFile = reactive({
  id: null,
  file_name: '',
  file_path: '',
  file_type: ''
})

// 删除状态
const deletingFileId = ref(null)

// 训练弹窗
const trainDialogVisible = ref(false)
const trainForm = reactive({
  epochs: 10,
  lr: 0.0001
})
const trainFormRef = ref(null)
const isTraining = ref(false)
const trainingFileId = ref(null)

// 测试弹窗
const testDialogVisible = ref(false)
const testForm = reactive({
  weight_path: ''
})
const testFormRef = ref(null)
const isTesting = ref(false)
const testingFileId = ref(null)

// 可用的权重文件列表（实际项目中可以从后端获取）
const availableWeights = ref([
  'models/weights/unet_octa.pth',
  'models/weights/unet_20260116_100530.pth'
])

// ==================== 生命周期钩子 ====================
onMounted(() => {
  fetchFileList()
})

// ==================== 核心功能函数 ====================

/**
 * 获取文件列表
 */
const fetchFileList = async () => {
  loading.value = true
  try {
    const params = {}
    if (filterType.value !== 'all') {
      params.file_type = filterType.value
    }
    
    const response = await axios.get(`${API_BASE_URL}/file/list`, { params })
    
    if (response.data.code === 200) {
      fileList.value = response.data.data || []
      ElMessage.success(`加载成功，共 ${fileList.value.length} 个文件`)
    } else {
      ElMessage.error(response.data.msg || '加载失败')
      fileList.value = []
    }
  } catch (error) {
    console.error('获取文件列表失败:', error)
    ElMessage.error('网络错误，请检查后端服务是否启动')
    fileList.value = []
  } finally {
    loading.value = false
  }
}

/**
 * 筛选条件改变
 */
const handleFilterChange = () => {
  fetchFileList()
}

/**
 * 删除文件
 */
const handleDelete = async (file) => {
  try {
    await ElMessageBox.confirm(
      `确定要删除文件"${file.file_name}"吗？此操作将同时删除数据库记录和本地文件，不可恢复！`,
      '删除确认',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )
    
    // 开始删除
    deletingFileId.value = file.id
    
    const response = await axios.delete(`${API_BASE_URL}/file/delete/${file.id}`)
    
    if (response.data.code === 200) {
      ElMessage.success('删除成功')
      // 刷新列表
      await fetchFileList()
    } else {
      ElMessage.error(response.data.msg || '删除失败')
    }
  } catch (error) {
    if (error !== 'cancel') {
      console.error('删除文件失败:', error)
      ElMessage.error('删除失败，请重试')
    }
  } finally {
    deletingFileId.value = null
  }
}

/**
 * 打开重新训练弹窗
 */
const handleReuse = (file) => {
  // 设置当前文件信息
  currentFile.id = file.id
  currentFile.file_name = file.file_name
  currentFile.file_path = file.file_path
  currentFile.file_type = file.file_type
  
  // 显示弹窗
  trainDialogVisible.value = true
}

/**
 * 确认训练
 */
const confirmTrain = async () => {
  // 参数验证
  if (trainForm.epochs <= 0 || trainForm.epochs > 100) {
    ElMessage.warning('训练轮数必须在1-100之间')
    return
  }
  if (trainForm.lr <= 0 || trainForm.lr > 0.01) {
    ElMessage.warning('学习率必须在0.00001-0.01之间')
    return
  }
  
  isTraining.value = true
  trainingFileId.value = currentFile.id
  
  try {
    const response = await axios.post(
      `${API_BASE_URL}/file/reuse/${currentFile.id}`,
      null,
      {
        params: {
          epochs: trainForm.epochs,
          lr: trainForm.lr
        }
      }
    )
    
    if (response.data.code === 200) {
      const result = response.data.data
      ElMessage.success(`训练成功！最终损失：${result.final_loss?.toFixed(4) || 'N/A'}`)
      
      // 关闭弹窗
      trainDialogVisible.value = false
      
      // 刷新列表
      await fetchFileList()
    } else {
      ElMessage.error(response.data.msg || '训练失败')
    }
  } catch (error) {
    console.error('训练失败:', error)
    ElMessage.error('训练失败，请检查后端服务或数据集格式')
  } finally {
    isTraining.value = false
    trainingFileId.value = null
  }
}

/**
 * 重置训练表单
 */
const resetTrainForm = () => {
  trainForm.epochs = 10
  trainForm.lr = 0.0001
  currentFile.id = null
  currentFile.file_name = ''
}

/**
 * 跳转到首页进行测试分割（重用历史图像）
 * 
 * 功能说明：
 * 1. 点击"测试分割"按钮后，跳转到首页
 * 2. 通过 query 参数传递 fileId
 * 3. 首页会自动加载该图像并预填充
 */
const handleTest = (file) => {
  // 使用 router.push 跳转到首页，携带 fileId 查询参数
  router.push({
    path: '/',
    query: {
      fileId: file.id
    }
  })
  
  // 显示提示消息
  ElMessage.info(`正在加载图像: ${file.file_name}`)
}

/**
 * 确认测试分割
 */
const confirmTest = async () => {
  isTesting.value = true
  testingFileId.value = currentFile.id
  
  try {
    const params = {}
    if (testForm.weight_path) {
      params.weight_path = testForm.weight_path
    }
    
    const response = await axios.post(
      `${API_BASE_URL}/file/test/${currentFile.id}`,
      null,
      { params }
    )
    
    if (response.data.code === 200) {
      const result = response.data.data
      ElMessage.success('分割成功！')
      
      // 可以选择打开结果图片或跳转到历史记录页面
      ElMessageBox.confirm(
        '分割完成，是否查看结果？',
        '提示',
        {
          confirmButtonText: '查看结果',
          cancelButtonText: '关闭',
          type: 'success'
        }
      ).then(() => {
        // 打开结果图片
        window.open(result.result_url, '_blank')
      }).catch(() => {
        // 用户选择关闭
      })
      
      // 关闭弹窗
      testDialogVisible.value = false
      
      // 刷新列表
      await fetchFileList()
    } else {
      ElMessage.error(response.data.msg || '分割失败')
    }
  } catch (error) {
    console.error('测试分割失败:', error)
    ElMessage.error('分割失败，请检查后端服务或图片格式')
  } finally {
    isTesting.value = false
    testingFileId.value = null
  }
}

/**
 * 重置测试表单
 */
const resetTestForm = () => {
  testForm.weight_path = ''
  currentFile.id = null
  currentFile.file_name = ''
}

// ==================== 工具函数 ====================

/**
 * 格式化文件大小
 */
const formatFileSize = (size) => {
  if (!size || size === 0) return '0 B'
  
  const units = ['B', 'KB', 'MB', 'GB']
  let unitIndex = 0
  let fileSize = size
  
  while (fileSize >= 1024 && unitIndex < units.length - 1) {
    fileSize /= 1024
    unitIndex++
  }
  
  return `${fileSize.toFixed(2)} ${units[unitIndex]}`
}

/**
 * 格式化上传时间
 */
const formatTime = (time) => {
  if (!time) return '-'
  
  // 如果是ISO格式时间戳
  if (time.includes('T')) {
    const date = new Date(time)
    return date.toLocaleString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit'
    }).replace(/\//g, '-')
  }
  
  // 如果已经是格式化的字符串，直接返回
  return time
}
</script>

<style scoped>
/* ==================== 容器样式 ==================== */
.file-manager-container {
  width: 80%;
  margin: 20px auto;
  padding-bottom: 40px;
}

/* ==================== 卡片通用样式 ==================== */
.card-container {
  margin-bottom: 20px;
  border-radius: 12px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

/* ==================== 标题卡片样式 ==================== */
.header-card {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.card-header {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.title-text {
  font-size: 28px;
  font-weight: bold;
  margin-bottom: 8px;
}

.subtitle-text {
  font-size: 14px;
  opacity: 0.9;
}

/* ==================== 筛选区域样式 ==================== */
.filter-section {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.filter-left {
  display: flex;
  gap: 12px;
  align-items: center;
}

.filter-right {
  display: flex;
  align-items: center;
}

.file-count {
  font-size: 14px;
  color: #606266;
  font-weight: 500;
}

/* ==================== 表格样式 ==================== */
.table-card {
  min-height: 400px;
}

.file-name {
  font-weight: 500;
  color: #303133;
}

.model-path {
  font-size: 12px;
  color: #409eff;
}

.no-model {
  font-size: 12px;
  color: #909399;
}

.action-buttons {
  display: flex;
  gap: 8px;
  justify-content: center;
}

/* ==================== 弹窗表单样式 ==================== */
.form-tip {
  font-size: 12px;
  color: #909399;
  margin-top: 4px;
}

.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
}

/* ==================== 响应式适配 ==================== */
@media (max-width: 1200px) {
  .file-manager-container {
    width: 90%;
  }
}

@media (max-width: 768px) {
  .file-manager-container {
    width: 95%;
  }
  
  .filter-section {
    flex-direction: column;
    gap: 12px;
    align-items: flex-start;
  }
  
  .filter-left {
    width: 100%;
    flex-direction: column;
  }
  
  .filter-left .el-select {
    width: 100% !important;
  }
  
  .action-buttons {
    flex-direction: column;
  }
}

/* ==================== Element Plus 组件样式覆盖 ==================== */
:deep(.el-table) {
  border-radius: 8px;
}

:deep(.el-button) {
  border-radius: 6px;
}

:deep(.el-select) {
  border-radius: 6px;
}

:deep(.el-dialog) {
  border-radius: 12px;
}

:deep(.el-dialog__header) {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border-radius: 12px 12px 0 0;
  padding: 20px;
}

:deep(.el-dialog__title) {
  color: white;
  font-weight: bold;
}

:deep(.el-dialog__headerbtn .el-dialog__close) {
  color: white;
}
</style>
