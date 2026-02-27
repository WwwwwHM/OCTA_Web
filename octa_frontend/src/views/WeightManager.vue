<template>
  <div class="weight-manager">
    <!-- Fix: 平台优化 - 权重上传管理界面 -->
    <el-card class="header-card">
      <template #header>
        <div class="card-header">
          <span class="title-text">模型权重管理</span>
          <span class="subtitle-text">上传和管理OCTA分割模型权重文件</span>
        </div>
      </template>
    </el-card>

    <!-- 权重上传区 -->
    <el-card class="upload-card">
      <h3 class="section-title">上传新权重文件</h3>
      
      <el-upload
        class="upload-demo"
        drag
        action=""
        :auto-upload="false"
        accept=".pth,.pt"
        :file-list="fileList"
        :before-upload="beforeUpload"
        @change="handleFileChange"
      >
        <el-icon class="el-icon--upload"><upload-filled /></el-icon>
        <div class="el-upload__text">
          将权重文件拖到此处，或<em>点击上传</em>
        </div>
        <template #tip>
          <div class="el-upload__tip">
            支持 .pth / .pt 格式，文件大小不超过200MB
          </div>
        </template>
      </el-upload>

      <div v-if="fileList.length" class="upload-actions">
        <el-button type="primary" :loading="uploading" @click="handleUpload">
          <el-icon><upload-filled /></el-icon>
          开始上传
        </el-button>
        <el-button @click="handleClear">清空</el-button>
      </div>
    </el-card>

    <!-- 权重列表区 -->
    <el-card class="list-card">
      <template #header>
        <div class="list-header">
          <h3>已上传权重列表</h3>
          <el-button type="primary" :icon="Refresh" @click="loadWeightList" :loading="loading">
            刷新列表
          </el-button>
        </div>
      </template>

      <el-table
        v-loading="loading"
        :data="weightList"
        stripe
        style="width: 100%"
        @selection-change="handleSelectionChange"
      >
        <el-table-column type="selection" width="55" />
        <el-table-column prop="file_name" label="文件名" min-width="200">
          <template #default="{ row }">
            <div class="file-name">
              <el-icon><document /></el-icon>
              <span>{{ row.file_name }}</span>
            </div>
          </template>
        </el-table-column>
        <el-table-column prop="model_type" label="模型类型" width="120">
          <template #default="{ row }">
            <el-tag v-if="row.model_type === 'unet'" type="success">U-Net</el-tag>
            <el-tag v-else-if="row.model_type === 'rs_unet3_plus'" type="warning">RS-Unet3+</el-tag>
            <el-tag v-else type="info">{{ row.model_type }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="file_size" label="文件大小" width="120">
          <template #default="{ row }">
            {{ formatFileSize(row.file_size) }}
          </template>
        </el-table-column>
        <el-table-column prop="upload_time" label="上传时间" width="180" />
        <el-table-column label="操作" width="180" fixed="right">
          <template #default="{ row }">
            <el-button
              type="primary"
              size="small"
              :icon="Download"
              @click="handleDownload(row)"
            >
              下载
            </el-button>
            <el-button
              type="danger"
              size="small"
              :icon="Delete"
              @click="handleDelete(row)"
            >
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>

      <div v-if="selectedWeights.length > 0" class="batch-actions">
        <el-button type="danger" :icon="Delete" @click="handleBatchDelete">
          批量删除 ({{ selectedWeights.length }})
        </el-button>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { UploadFilled, Refresh, Download, Delete, Document } from '@element-plus/icons-vue'
import axios from 'axios'

// 响应式数据
const fileList = ref([])
const weightList = ref([])
const selectedWeights = ref([])
const uploading = ref(false)
const loading = ref(false)

// 文件变化处理
const handleFileChange = (file, files) => {
  fileList.value = files
}

// 上传前校验
const beforeUpload = (file) => {
  const isValidFormat = file.name.endsWith('.pth') || file.name.endsWith('.pt')
  const isValidSize = file.size / 1024 / 1024 < 200

  if (!isValidFormat) {
    ElMessage.error('仅支持 .pth 或 .pt 格式的权重文件！')
    return false
  }
  if (!isValidSize) {
    ElMessage.error('文件大小不能超过 200MB！')
    return false
  }
  return true
}

// 上传权重
const handleUpload = async () => {
  if (!fileList.value.length) {
    ElMessage.warning('请先选择权重文件')
    return
  }

  uploading.value = true
  
  try {
    const formData = new FormData()
    formData.append('file', fileList.value[0].raw)
    
    const response = await axios.post(
      'http://127.0.0.1:8000/api/v1/weight/upload',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        timeout: 60000
      }
    )

    if (response.data.code === 200) {
      ElMessage.success('权重上传成功！')
      fileList.value = []
      await loadWeightList()
    } else {
      ElMessage.error(response.data.msg || '上传失败')
    }
  } catch (error) {
    console.error('上传失败:', error)
    ElMessage.error(error.response?.data?.detail || '上传失败，请检查网络连接')
  } finally {
    uploading.value = false
  }
}

// 清空文件列表
const handleClear = () => {
  fileList.value = []
}

// 加载权重列表
const loadWeightList = async () => {
  loading.value = true
  
  try {
    const response = await axios.get('http://127.0.0.1:8000/api/v1/weight/list')
    
    if (response.data.code === 200) {
      weightList.value = response.data.data || []
    } else {
      ElMessage.error(response.data.msg || '加载失败')
    }
  } catch (error) {
    console.error('加载权重列表失败:', error)
    ElMessage.error('加载失败，请检查网络连接')
  } finally {
    loading.value = false
  }
}

// 格式化文件大小
const formatFileSize = (bytes) => {
  if (!bytes) return '未知'
  const mb = bytes / 1024 / 1024
  return mb >= 1 ? `${mb.toFixed(2)} MB` : `${(bytes / 1024).toFixed(2)} KB`
}

// 下载权重
const handleDownload = (row) => {
  window.open(`http://127.0.0.1:8000${row.file_path}`, '_blank')
}

// 删除单个权重
const handleDelete = async (row) => {
  try {
    await ElMessageBox.confirm(
      `确定要删除权重文件 "${row.file_name}" 吗？此操作不可恢复。`,
      '删除确认',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )

    const response = await axios.delete(
      `http://127.0.0.1:8000/api/v1/weight/delete/${row.weight_id}`
    )

    if (response.data.code === 200) {
      ElMessage.success('删除成功')
      await loadWeightList()
    } else {
      ElMessage.error(response.data.msg || '删除失败')
    }
  } catch (error) {
    if (error !== 'cancel') {
      console.error('删除失败:', error)
      ElMessage.error(error.response?.data?.detail || '删除失败')
    }
  }
}

// 选择变化
const handleSelectionChange = (selection) => {
  selectedWeights.value = selection
}

// 批量删除
const handleBatchDelete = async () => {
  try {
    await ElMessageBox.confirm(
      `确定要删除选中的 ${selectedWeights.value.length} 个权重文件吗？此操作不可恢复。`,
      '批量删除确认',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )

    const deletePromises = selectedWeights.value.map(row =>
      axios.delete(`http://127.0.0.1:8000/api/v1/weight/delete/${row.weight_id}`)
    )

    await Promise.all(deletePromises)
    
    ElMessage.success('批量删除成功')
    await loadWeightList()
  } catch (error) {
    if (error !== 'cancel') {
      console.error('批量删除失败:', error)
      ElMessage.error('批量删除失败')
    }
  }
}

// 组件挂载时加载列表
onMounted(() => {
  loadWeightList()
})
</script>

<style scoped>
.weight-manager {
  padding: 24px;
  max-width: 1400px;
  margin: 0 auto;
}

.header-card {
  margin-bottom: 24px;
}

.card-header {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.title-text {
  font-size: 24px;
  font-weight: 600;
  color: #303133;
}

.subtitle-text {
  font-size: 14px;
  color: #909399;
}

.upload-card,
.list-card {
  margin-bottom: 24px;
}

.section-title {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 16px;
  color: #303133;
}

.upload-demo {
  margin-bottom: 16px;
}

.upload-actions {
  display: flex;
  gap: 12px;
  justify-content: center;
  margin-top: 16px;
}

.list-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.list-header h3 {
  margin: 0;
  font-size: 16px;
  font-weight: 600;
  color: #303133;
}

.file-name {
  display: flex;
  align-items: center;
  gap: 8px;
}

.batch-actions {
  margin-top: 16px;
  padding-top: 16px;
  border-top: 1px solid #ebeef5;
}

/* 响应式 */
@media (max-width: 768px) {
  .weight-manager {
    padding: 16px;
  }

  .title-text {
    font-size: 20px;
  }
}
</style>
