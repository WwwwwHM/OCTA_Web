<template>
  <div class="file-selector">
    <!-- 标签页切换 -->
    <el-tabs v-model="activeTab" class="file-selector-tabs">
      <!-- 上传新文件标签页 -->
      <el-tab-pane label="上传新文件" name="upload">
        <slot name="upload-content"></slot>
      </el-tab-pane>

      <!-- 选择已上传文件标签页 -->
      <el-tab-pane label="选择已上传文件" name="select">
        <div class="select-tab-content">
          <!-- 加载指示器 -->
          <el-skeleton v-if="loading" :rows="5" animated />

          <!-- 空文件提示 -->
          <el-empty 
            v-else-if="fileList.length === 0"
            description="暂无已上传的文件"
            :image-size="100"
          />

          <!-- 文件列表表格 -->
          <el-table
            v-else
            :data="fileList"
            stripe
            style="width: 100%"
            :default-sort="{ prop: 'upload_time', order: 'descending' }"
            class="file-table"
          >
            <!-- 文件名列 -->
            <el-table-column
              prop="file_name"
              label="文件名"
              min-width="150"
              show-overflow-tooltip
            />

            <!-- 文件大小列 -->
            <el-table-column
              prop="file_size"
              label="文件大小"
              width="100"
              align="center"
            >
              <template #default="{ row }">
                {{ formatFileSize(row.file_size) }}
              </template>
            </el-table-column>

            <!-- 上传时间列 -->
            <el-table-column
              prop="upload_time"
              label="上传时间"
              width="180"
              align="center"
              sortable
            />

            <!-- 操作列 -->
            <el-table-column
              label="操作"
              width="120"
              align="center"
              fixed="right"
            >
              <template #default="{ row }">
                <!-- 对于图片：显示选择和预览按钮 -->
                <template v-if="fileType === 'image'">
                  <el-button
                    type="primary"
                    size="small"
                    @click="handleSelectImage(row)"
                  >
                    选择
                  </el-button>
                </template>

                <!-- 对于数据集：显示选择按钮 -->
                <template v-else-if="fileType === 'dataset'">
                  <el-button
                    :type="selectedFile?.id === row.id ? 'success' : 'primary'"
                    size="small"
                    @click="handleSelectDataset(row)"
                  >
                    {{ selectedFile?.id === row.id ? '已选择' : '选择' }}
                  </el-button>
                </template>
              </template>
            </el-table-column>
          </el-table>

          <!-- 已选择的文件展示 (仅数据集) -->
          <div v-if="fileType === 'dataset' && selectedFile" class="selected-dataset-info">
            <el-alert
              type="success"
              :closable="false"
              show-icon
            >
              <template #default>
                <strong>✓ 已选择数据集：</strong> {{ selectedFile.file_name }}
              </template>
            </el-alert>
          </div>

          <!-- 已选择的图片预览 (仅图片) -->
          <div v-if="fileType === 'image' && previewUrl" class="selected-image-preview">
            <div class="preview-label">图片预览</div>
            <img :src="previewUrl" alt="预览图片" class="preview-image">
            <div class="preview-info">
              <p><strong>文件名：</strong> {{ selectedFile?.file_name }}</p>
              <p><strong>大小：</strong> {{ formatFileSize(selectedFile?.file_size) }}</p>
              <p><strong>上传时间：</strong> {{ selectedFile?.upload_time }}</p>
            </div>
          </div>
        </div>
      </el-tab-pane>
    </el-tabs>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import { ElMessage } from 'element-plus'
import axios from 'axios'

// ==================== Props ====================

const props = defineProps({
  // 文件类型：'image' 或 'dataset'
  fileType: {
    type: String,
    required: true,
    validator: (value) => ['image', 'dataset'].includes(value)
  },
  
  // API基础URL
  apiBaseUrl: {
    type: String,
    default: 'http://127.0.0.1:8000'
  }
})

// ==================== Emits ====================

const emit = defineEmits([
  'file-selected', // 文件选择完成
  'preview-loaded' // 预览加载完成（图片）
])

// ==================== 响应式数据 ====================

const activeTab = ref('upload')
const fileList = ref([])
const loading = ref(false)
const selectedFile = ref(null)
const previewUrl = ref(null)

// ==================== 计算属性 ====================

// 获取标签文本
const tabLabel = computed(() => {
  return props.fileType === 'image' ? '图片文件' : '数据集文件'
})

// ==================== 方法 ====================

/**
 * 格式化文件大小
 */
const formatFileSize = (bytes) => {
  if (!bytes) return '0 B'
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(1024))
  return Math.round((bytes / Math.pow(1024, i)) * 100) / 100 + ' ' + sizes[i]
}

/**
 * 加载已上传文件列表
 */
const loadFileList = async () => {
  loading.value = true
  try {
    const response = await axios.get(
      `${props.apiBaseUrl}/file/list`,
      { params: { file_type: props.fileType } }
    )

    if (response.data?.code === 200) {
      fileList.value = response.data.data || []
      if (fileList.value.length === 0) {
        ElMessage.info(`暂无已上传的${props.fileType === 'image' ? '图片' : '数据集'}`)
      }
    } else {
      ElMessage.error(response.data?.msg || '加载文件列表失败')
    }
  } catch (error) {
    console.error('加载文件列表失败:', error)
    ElMessage.error('加载文件列表失败，请检查网络连接')
  } finally {
    loading.value = false
  }
}

/**
 * 选择图片并加载预览
 */
const handleSelectImage = async (file) => {
  selectedFile.value = file
  loading.value = true

  try {
    const response = await axios.get(
      `${props.apiBaseUrl}/file/preview/${file.id}`
    )

    if (response.data?.code === 200) {
      previewUrl.value = response.data.data.preview_url
      ElMessage.success('图片预览加载成功')
      
      // 发送选择完成事件
      emit('file-selected', {
        id: file.id,
        ...file
      })
      emit('preview-loaded', {
        id: file.id,
        previewUrl: previewUrl.value
      })
    } else {
      ElMessage.error(response.data?.msg || '加载预览失败')
    }
  } catch (error) {
    console.error('加载预览失败:', error)
    ElMessage.error('加载预览失败，请稍后重试')
  } finally {
    loading.value = false
  }
}

/**
 * 选择数据集
 */
const handleSelectDataset = (file) => {
  selectedFile.value = file
  ElMessage.success(`已选择数据集：${file.file_name}`)
  
  // 发送选择完成事件
  emit('file-selected', {
    id: file.id,
    ...file
  })
}

/**
 * 监听标签页切换
 */
watch(
  () => activeTab.value,
  (newTab) => {
    if (newTab === 'select') {
      // 切换到"选择已上传文件"标签页时，加载文件列表
      loadFileList()
    }
  }
)

// ==================== 对外暴露的方法 ====================

defineExpose({
  loadFileList,
  selectedFile,
  previewUrl,
  activeTab
})
</script>

<style scoped>
.file-selector {
  width: 100%;
}

.file-selector-tabs :deep(.el-tabs__content) {
  padding: 0;
}

.select-tab-content {
  padding: 20px 0;
}

.file-table {
  margin: 20px 0;
}

.selected-dataset-info {
  margin-top: 20px;
  padding: 15px;
  background-color: #f0f9ff;
  border-radius: 4px;
}

.selected-image-preview {
  margin-top: 20px;
  padding: 15px;
  background-color: #f5f7fa;
  border-radius: 4px;
  text-align: center;
}

.preview-label {
  font-weight: bold;
  color: #303133;
  margin-bottom: 10px;
  font-size: 14px;
}

.preview-image {
  max-width: 100%;
  max-height: 400px;
  border-radius: 4px;
  border: 1px solid #dcdfe6;
  margin: 10px 0;
}

.preview-info {
  margin-top: 15px;
  text-align: left;
  color: #606266;
  font-size: 13px;
}

.preview-info p {
  margin: 5px 0;
  line-height: 1.6;
}
</style>
