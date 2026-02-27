<template>
  <div class="page-wrap">
    <el-card class="panel" shadow="never">
      <template #header>
        <div class="panel-title">OCTA 血管分割平台（模型即插即用）</div>
      </template>

      <div class="section-grid">
        <!-- 模型选择区 -->
        <div class="section-block">
          <h3>1. 模型选择</h3>
          <el-select
            v-model="selectedModel"
            class="full-width"
            placeholder="请选择模型"
            @change="handleModelChange"
            :disabled="isModelListEmpty"
          >
            <el-option
              v-for="model in modelList"
              :key="model.name"
              :label="`${model.name}（${formatInputSize(model.input_size)}）`"
              :value="model.name"
            />
          </el-select>
          <div v-if="currentModelConfig" class="tip-text">
            当前模型：{{ currentModelConfig.name }}，输入尺寸：{{ formatInputSize(currentModelConfig.input_size) }}，
            类别数：{{ currentModelConfig.num_classes }}
          </div>
          <div v-else class="tip-text">暂无可用模型，请联系管理员</div>
        </div>

        <!-- 图片上传区 -->
        <div class="section-block">
          <h3>2. 图片上传与缩放预览</h3>
          <el-upload
            class="full-width"
            drag
            action=""
            :auto-upload="false"
            :show-file-list="false"
            accept=".jpg,.jpeg,.png,.bmp,.JPG,.JPEG,.PNG,.BMP"
            @change="handleImageUpload"
          >
            <el-icon class="upload-icon"><UploadFilled /></el-icon>
            <div class="el-upload__text">拖拽或点击上传（仅支持 jpg/png/bmp）</div>
          </el-upload>

          <div class="preview-grid">
            <div class="image-box" v-if="originalPreviewUrl">
              <div class="box-title">原图预览</div>
              <img :src="originalPreviewUrl" alt="原图" />
            </div>
            <div class="image-box" v-if="processedPreviewUrl">
              <div class="box-title">按模型尺寸处理后预览</div>
              <img :src="processedPreviewUrl" alt="处理后图像" />
              <div class="tip-text">
                当前模型输入尺寸：{{ formatInputSize(currentModelConfig?.input_size) }}，已自动缩放图片
              </div>
            </div>
          </div>
        </div>

        <!-- 推理操作区 -->
        <div class="section-block">
          <h3>3. 推理操作</h3>
          <div class="action-row">
            <el-input
              v-model="weightId"
              placeholder="可选：weight_id（不填使用模型默认权重）"
              clearable
            />
            <div class="action-buttons">
              <el-button
                type="primary"
                :loading="isPredicting"
                :disabled="!canPredict"
                @click="handlePredict"
              >
                {{ isPredicting ? '推理中...' : '开始推理' }}
              </el-button>
              <el-button @click="handleReset">重置</el-button>
            </div>
          </div>
          <div class="tip-text" v-if="inferTime !== null">
            推理耗时：{{ inferTime }}s
          </div>
        </div>

        <!-- 结果展示区 -->
        <div class="section-block">
          <h3>4. 分割结果</h3>
          <div class="compare-grid" v-if="maskPreviewUrl">
            <div class="image-box">
              <div class="box-title">原图</div>
              <img :src="originalPreviewUrl" alt="原图" />
            </div>
            <div class="image-box">
              <div class="box-title">Mask 结果（Base64）</div>
              <img :src="maskPreviewUrl" alt="mask结果" />
            </div>
          </div>
          <el-empty v-else description="暂无分割结果" />
        </div>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { UploadFilled } from '@element-plus/icons-vue'
import axios from 'axios'

const API_BASE_URL = 'http://127.0.0.1:8000'
const API_SEG_BASE = `${API_BASE_URL}/api/v1/seg`

const modelList = ref([])
const selectedModel = ref('')
const currentModelConfig = ref(null)

const weightId = ref('')
const uploadedFile = ref(null)
const scaledImageFile = ref(null)

const originalPreviewUrl = ref('')
const processedPreviewUrl = ref('')
const maskPreviewUrl = ref('')

const isPredicting = ref(false)
const inferTime = ref(null)

const isModelListEmpty = computed(() => modelList.value.length === 0)

const canPredict = computed(() => {
  return Boolean(
    selectedModel.value &&
      scaledImageFile.value &&
      !isPredicting.value &&
      !isModelListEmpty.value
  )
})

const formatInputSize = (inputSize) => {
  if (!Array.isArray(inputSize) || inputSize.length !== 2) return '-'
  return `${inputSize[0]}×${inputSize[1]}`
}

/**
 * 页面初始化：拉取后端已注册模型列表
 * 并默认选择第一个模型。
 */
const fetchModelList = async () => {
  try {
    const response = await axios.get(`${API_SEG_BASE}/models`, { timeout: 5000 })
    const payload = response?.data

    if (payload?.code !== 200 || !Array.isArray(payload?.data)) {
      throw new Error(payload?.msg || '模型列表格式错误')
    }

    modelList.value = payload.data
    if (modelList.value.length === 0) {
      // 异常场景②：模型列表为空
      currentModelConfig.value = null
      selectedModel.value = ''
      ElMessage.warning('暂无可用模型，请联系管理员')
      return
    }

    selectedModel.value = modelList.value[0].name
    currentModelConfig.value = modelList.value[0]
  } catch (error) {
    // 异常场景①：/models接口调用失败
    ElMessage.error('加载模型列表失败，请刷新页面重试')
  }
}

/**
 * 切换模型：更新配置并自动清空旧图像和旧结果
 */
const handleModelChange = (modelName) => {
  const found = modelList.value.find((item) => item.name === modelName)
  if (!found) {
    ElMessage.error('模型未找到，请刷新页面重试')
    return
  }

  currentModelConfig.value = found
  clearImageAndResult()
}

/**
 * 清空上传图片与分割结果（模型切换时调用）
 */
const clearImageAndResult = () => {
  uploadedFile.value = null
  scaledImageFile.value = null
  originalPreviewUrl.value = ''
  processedPreviewUrl.value = ''
  maskPreviewUrl.value = ''
  inferTime.value = null
}

/**
 * 图片格式前置校验：仅允许jpg/png/bmp
 */
const validateImageFormat = (file) => {
  const fileName = (file?.name || '').toLowerCase()
  const allowedExt = ['.jpg', '.jpeg', '.png', '.bmp']
  const isValid = allowedExt.some((ext) => fileName.endsWith(ext))

  if (!isValid) {
    // 异常场景⑤：图片格式错误
    ElMessage.error('仅支持jpg/png/bmp格式图片')
  }

  return isValid
}

/**
 * 关键逻辑：按当前模型input_size缩放图片，并返回新的File对象和预览URL
 */
const resizeImageByModelConfig = (file, inputSize) => {
  return new Promise((resolve, reject) => {
    try {
      const [targetWidth, targetHeight] = inputSize
      const reader = new FileReader()

      reader.onload = () => {
        const image = new Image()
        image.onload = () => {
          const canvas = document.createElement('canvas')
          canvas.width = targetWidth
          canvas.height = targetHeight

          const ctx = canvas.getContext('2d')
          ctx.drawImage(image, 0, 0, targetWidth, targetHeight)

          canvas.toBlob(
            (blob) => {
              if (!blob) {
                reject(new Error('图片缩放失败：无法生成Blob'))
                return
              }

              const scaledFile = new File([blob], file.name, { type: 'image/png' })
              const previewUrl = URL.createObjectURL(blob)
              resolve({ scaledFile, previewUrl })
            },
            'image/png'
          )
        }
        image.onerror = () => reject(new Error('图片读取失败：无法加载图像'))
        image.src = reader.result
      }

      reader.onerror = () => reject(new Error('图片读取失败：FileReader异常'))
      reader.readAsDataURL(file)
    } catch (error) {
      reject(error)
    }
  })
}

/**
 * 上传后处理：校验格式 + 生成原图预览 + 缩放图预览和File对象
 */
const handleImageUpload = async (uploadFile) => {
  try {
    const rawFile = uploadFile?.raw
    if (!rawFile) {
      ElMessage.error('上传失败：未读取到文件')
      return
    }

    if (!currentModelConfig.value) {
      ElMessage.error('请先选择模型')
      return
    }

    if (!validateImageFormat(rawFile)) {
      return
    }

    uploadedFile.value = rawFile
    originalPreviewUrl.value = URL.createObjectURL(rawFile)

    const { scaledFile, previewUrl } = await resizeImageByModelConfig(
      rawFile,
      currentModelConfig.value.input_size
    )

    scaledImageFile.value = scaledFile
    processedPreviewUrl.value = previewUrl
    maskPreviewUrl.value = ''
    inferTime.value = null
  } catch (error) {
    ElMessage.error(`图片处理失败：${error.message}`)
  }
}

/**
 * 关键逻辑：调用后端推理接口
 * FormData字段：model_name/weight_id/image_file
 */
const handlePredict = async () => {
  if (isModelListEmpty.value) {
    ElMessage.warning('暂无可用模型，请联系管理员')
    return
  }

  if (!selectedModel.value) {
    ElMessage.warning('请先选择模型')
    return
  }

  // 异常场景③：图片上传为空
  if (!scaledImageFile.value) {
    ElMessage.warning('请先上传图片')
    return
  }

  try {
    isPredicting.value = true

    const formData = new FormData()
    formData.append('model_name', selectedModel.value)
    formData.append('weight_id', weightId.value || '')
    formData.append('image_file', scaledImageFile.value)

    const response = await axios.post(`${API_SEG_BASE}/predict`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 5000
    })

    const payload = response?.data
    if (payload?.code !== 200) {
      throw new Error(payload?.msg || '推理失败')
    }

    const data = payload?.data || {}
    if (!data.mask_base64) {
      throw new Error('后端未返回mask_base64')
    }

    maskPreviewUrl.value = `data:image/png;base64,${data.mask_base64}`
    inferTime.value = data.infer_time ?? null
    ElMessage.success('推理成功')
  } catch (error) {
    // 异常场景⑥：接口超时
    if (error?.code === 'ECONNABORTED') {
      ElMessage.error('推理超时，请检查网络或重试')
      return
    }

    // 异常场景④：接口错误码，优先使用后端msg
    const backendMsg = error?.response?.data?.msg
    ElMessage.error(backendMsg || `推理失败：${error.message}`)
  } finally {
    isPredicting.value = false
  }
}

/**
 * 体验优化：重置按钮清空所有输入和结果
 */
const handleReset = () => {
  selectedModel.value = modelList.value[0]?.name || ''
  currentModelConfig.value = modelList.value[0] || null
  weightId.value = ''
  clearImageAndResult()
}

onMounted(async () => {
  await fetchModelList()
})
</script>

<style scoped>
.page-wrap {
  max-width: 1200px;
  margin: 24px auto;
  padding: 0 16px;
}

.panel {
  border-radius: 10px;
}

.panel-title {
  font-size: 20px;
  font-weight: 700;
}

.section-grid {
  display: grid;
  gap: 16px;
}

.section-block {
  border: 1px solid #ebeef5;
  border-radius: 10px;
  padding: 16px;
}

.section-block h3 {
  margin: 0 0 12px;
  font-size: 16px;
}

.full-width {
  width: 100%;
}

.tip-text {
  margin-top: 8px;
  color: #606266;
  font-size: 13px;
}

.upload-icon {
  font-size: 28px;
  color: #909399;
}

.preview-grid {
  margin-top: 14px;
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
}

.image-box {
  border: 1px dashed #dcdfe6;
  border-radius: 8px;
  padding: 10px;
  text-align: center;
}

.box-title {
  font-size: 13px;
  margin-bottom: 8px;
  color: #606266;
}

.image-box img {
  width: 100%;
  max-height: 320px;
  object-fit: contain;
  border-radius: 6px;
}

.action-row {
  display: grid;
  gap: 10px;
  grid-template-columns: 1fr auto;
  align-items: center;
}

.action-buttons {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.compare-grid {
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
}

@media (max-width: 768px) {
  .action-row {
    grid-template-columns: 1fr;
  }
}
</style>
