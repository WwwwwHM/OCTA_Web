<template>
  <el-dialog
    :model-value="visible"
    title="RS-Unet3+ 训练参数配置（OCTA专用）"
    width="800px"
    :before-close="handleCancel"
    destroy-on-close
  >
    <el-form
      ref="formRef"
      :model="formData"
      :rules="rules"
      label-width="120px"
      :inline="true"
      class="param-form"
    >
      <!-- 训练轮数 -->
      <el-form-item label="训练轮数" prop="epochs" class="form-item-full">
        <el-tooltip
          content="RS-Unet3+ 最优 epochs=200，适配 OCTA 小数据集，充分学习血管细节特征"
          placement="top"
        >
          <el-input-number
            v-model="formData.epochs"
            :min="50"
            :max="500"
            :step="10"
            controls-position="right"
            style="width: 200px"
          />
        </el-tooltip>
        <span class="param-tip">推荐值：200（OCTA血管分割最佳）</span>
      </el-form-item>

      <!-- 学习率 -->
      <el-form-item label="学习率" prop="lr" class="form-item-full">
        <el-tooltip
          content="推荐 1e-4（0.0001），过大易震荡，过小收敛慢。支持科学计数法（如 1e-4）或小数（0.0001）"
          placement="top"
        >
          <el-input
            v-model="formData.lr"
            placeholder="如：1e-4 或 0.0001"
            style="width: 200px"
            @blur="validateLearningRate"
          />
        </el-tooltip>
        <span class="param-tip">推荐值：1e-4（AdamW优化器）</span>
      </el-form-item>

      <!-- 权重衰减 -->
      <el-form-item label="权重衰减" prop="weight_decay" class="form-item-full">
        <el-tooltip
          content="L2正则化系数，推荐 1e-5，防止过拟合。支持科学计数法（如 1e-5）或小数（0.00001）"
          placement="top"
        >
          <el-input
            v-model="formData.weight_decay"
            placeholder="如：1e-5 或 0.00001"
            style="width: 200px"
            @blur="validateWeightDecay"
          />
        </el-tooltip>
        <span class="param-tip">推荐值：1e-5（平衡泛化与拟合）</span>
      </el-form-item>

      <!-- 损失函数 -->
      <el-form-item label="损失函数" prop="loss_function" class="form-item-full">
        <el-tooltip
          content="Lovasz-Softmax 擅长处理血管像素不平衡，交叉熵提供像素级监督，两者联合效果最佳"
          placement="top"
        >
          <el-select
            v-model="formData.loss_function"
            style="width: 200px"
            disabled
          >
            <el-option
              label="Lovasz-Softmax + 交叉熵"
              value="lovasz_ce"
            />
            <el-option
              label="Dice Loss + BCE"
              value="dice_bce"
            />
          </el-select>
        </el-tooltip>
        <span class="param-tip">✅ 固定方案（OCTA血管最优）</span>
      </el-form-item>

      <!-- 学习率调度 -->
      <el-form-item label="学习率调度" prop="lr_scheduler" class="form-item-full">
        <el-tooltip
          content="余弦退火调度器，从初始学习率平滑降至0，避免后期震荡，适合长训练周期"
          placement="top"
        >
          <el-select
            v-model="formData.lr_scheduler"
            style="width: 200px"
            disabled
          >
            <el-option
              label="余弦退火（CosineAnnealingLR）"
              value="cosine"
            />
            <el-option
              label="步进衰减（StepLR）"
              value="step"
            />
          </el-select>
        </el-tooltip>
        <span class="param-tip">✅ 固定方案（200轮最优）</span>
      </el-form-item>

      <!-- 批量大小 -->
      <el-form-item label="批量大小" prop="batch_size" class="form-item-full">
        <el-tooltip
          content="批量大小建议 4（8GB显存）或 2（4GB显存/CPU）。过大可能OOM，过小训练不稳定"
          placement="top"
        >
          <el-input-number
            v-model="formData.batch_size"
            :min="1"
            :max="8"
            :step="1"
            controls-position="right"
            style="width: 200px"
          />
        </el-tooltip>
        <span class="param-tip">推荐值：4（GPU 8GB）或 2（CPU）</span>
      </el-form-item>
    </el-form>

    <!-- 参数配置预览 -->
    <el-alert
      type="info"
      :closable="false"
      style="margin-top: 20px"
    >
      <template #title>
        <strong>当前配置预览</strong>
      </template>
      <div class="config-preview">
        <p>
          <strong>训练轮数：</strong>{{ formData.epochs }} 轮
          <span v-if="formData.epochs >= 200" class="badge-success">✓ 推荐</span>
          <span v-else class="badge-warning">⚠ 建议≥200</span>
        </p>
        <p>
          <strong>学习率：</strong>{{ formData.lr }}
          <span v-if="isValidLR" class="badge-success">✓ 格式正确</span>
          <span v-else class="badge-error">✗ 格式错误</span>
        </p>
        <p>
          <strong>权重衰减：</strong>{{ formData.weight_decay }}
          <span v-if="isValidWD" class="badge-success">✓ 格式正确</span>
          <span v-else class="badge-error">✗ 格式错误</span>
        </p>
        <p><strong>损失函数：</strong>Lovasz-Softmax + 交叉熵（联合损失）</p>
        <p><strong>学习率调度：</strong>余弦退火（{{ formData.epochs }} 轮平滑衰减）</p>
        <p><strong>批量大小：</strong>{{ formData.batch_size }}</p>
        <p class="preview-note">
          💡 预计训练时间：{{ estimateTrainingTime }} 
          （基于 {{ formData.batch_size }} batch_size，CPU模式）
        </p>
      </div>
    </el-alert>

    <!-- 操作按钮 -->
    <template #footer>
      <div class="dialog-footer">
        <el-button @click="handleCancel">取消</el-button>
        <el-button
          type="primary"
          :disabled="!isFormValid"
          @click="handleConfirm"
        >
          确认配置
        </el-button>
      </div>
    </template>
  </el-dialog>
</template>

<script setup>
import { ref, reactive, computed, watch } from 'vue'
import { ElMessage } from 'element-plus'

// ==================== Props & Emits ====================

const props = defineProps({
  visible: {
    type: Boolean,
    default: false
  },
  defaultParams: {
    type: Object,
    default: () => ({
      epochs: 200,
      lr: '1e-4',
      weight_decay: '1e-5',
      loss_function: 'lovasz_ce',
      lr_scheduler: 'cosine',
      batch_size: 4
    })
  }
})

const emit = defineEmits(['confirm', 'cancel'])

// ==================== 响应式数据 ====================

const formRef = ref(null)
const formData = reactive({
  epochs: 200,
  lr: '1e-4',
  weight_decay: '1e-5',
  loss_function: 'lovasz_ce',
  lr_scheduler: 'cosine',
  batch_size: 4
})

// 表单校验规则
const rules = {
  epochs: [
    { required: true, message: '请输入训练轮数', trigger: 'blur' },
    { type: 'number', min: 50, max: 500, message: '轮数范围：50-500', trigger: 'blur' }
  ],
  lr: [
    { required: true, message: '请输入学习率', trigger: 'blur' },
    { validator: validateLRFormat, trigger: 'blur' }
  ],
  weight_decay: [
    { required: true, message: '请输入权重衰减', trigger: 'blur' },
    { validator: validateWDFormat, trigger: 'blur' }
  ],
  batch_size: [
    { required: true, message: '请输入批量大小', trigger: 'blur' },
    { type: 'number', min: 1, max: 8, message: '批量大小范围：1-8', trigger: 'blur' }
  ]
}

// ==================== 计算属性 ====================

// 学习率格式校验
const isValidLR = computed(() => {
  const lr = formData.lr.toString().trim()
  // 支持科学计数法（如 1e-4）或小数（如 0.0001）
  const scientificPattern = /^[0-9.]+e-?[0-9]+$/i
  const decimalPattern = /^0\.\d+$/
  if (!scientificPattern.test(lr) && !decimalPattern.test(lr)) {
    return false
  }
  const value = parseFloat(lr)
  return value >= 1e-5 && value <= 1e-3
})

// 权重衰减格式校验
const isValidWD = computed(() => {
  const wd = formData.weight_decay.toString().trim()
  const scientificPattern = /^[0-9.]+e-?[0-9]+$/i
  const decimalPattern = /^0\.\d+$/
  if (!scientificPattern.test(wd) && !decimalPattern.test(wd)) {
    return false
  }
  const value = parseFloat(wd)
  return value >= 1e-6 && value <= 1e-4
})

// 表单整体有效性
const isFormValid = computed(() => {
  return (
    formData.epochs >= 50 &&
    formData.epochs <= 500 &&
    isValidLR.value &&
    isValidWD.value &&
    formData.batch_size >= 1 &&
    formData.batch_size <= 8
  )
})

// 预计训练时间
const estimateTrainingTime = computed(() => {
  const baseTime = 0.5 // 每轮基准时间（分钟/epoch，基于CPU）
  const totalMinutes = formData.epochs * baseTime * (4 / formData.batch_size)
  const hours = Math.floor(totalMinutes / 60)
  const minutes = Math.round(totalMinutes % 60)
  return hours > 0 ? `约 ${hours} 小时 ${minutes} 分钟` : `约 ${minutes} 分钟`
})

// ==================== 校验函数 ====================

function validateLRFormat(rule, value, callback) {
  if (!isValidLR.value) {
    callback(new Error('学习率格式错误或超出范围 [1e-5, 1e-3]'))
  } else {
    callback()
  }
}

function validateWDFormat(rule, value, callback) {
  if (!isValidWD.value) {
    callback(new Error('权重衰减格式错误或超出范围 [1e-6, 1e-4]'))
  } else {
    callback()
  }
}

function validateLearningRate() {
  if (!isValidLR.value) {
    ElMessage.warning('学习率格式错误！请输入科学计数法（如 1e-4）或小数（如 0.0001），范围 [1e-5, 1e-3]')
  }
}

function validateWeightDecay() {
  if (!isValidWD.value) {
    ElMessage.warning('权重衰减格式错误！请输入科学计数法（如 1e-5）或小数（如 0.00001），范围 [1e-6, 1e-4]')
  }
}

// ==================== 事件处理 ====================

const handleConfirm = async () => {
  if (!formRef.value) return

  try {
    await formRef.value.validate()
    
    // 转换学习率和权重衰减为数值
    const params = {
      epochs: formData.epochs,
      lr: parseFloat(formData.lr),
      weight_decay: parseFloat(formData.weight_decay),
      loss_function: formData.loss_function,
      lr_scheduler: formData.lr_scheduler,
      batch_size: formData.batch_size
    }
    
    emit('confirm', params)
    ElMessage.success('参数配置成功！')
  } catch (error) {
    ElMessage.error('请检查表单输入是否正确')
  }
}

const handleCancel = () => {
  resetForm()
  emit('cancel')
}

const resetForm = () => {
  Object.assign(formData, props.defaultParams)
}

// ==================== 监听器 ====================

// 监听弹窗显隐，显示时重置表单
watch(() => props.visible, (newVal) => {
  if (newVal) {
    resetForm()
  }
})

// 监听默认参数变化
watch(() => props.defaultParams, (newVal) => {
  if (newVal) {
    Object.assign(formData, newVal)
  }
}, { deep: true })
</script>

<style scoped>
.param-form {
  padding: 10px 0;
}

.form-item-full {
  width: 100%;
  margin-bottom: 20px;
}

.form-item-full :deep(.el-form-item__content) {
  display: flex;
  align-items: center;
  gap: 15px;
}

.param-tip {
  color: #909399;
  font-size: 12px;
  margin-left: 10px;
}

.config-preview {
  padding: 10px 0;
  line-height: 1.8;
}

.config-preview p {
  margin: 8px 0;
  font-size: 14px;
}

.config-preview strong {
  color: #303133;
  font-weight: 600;
  min-width: 100px;
  display: inline-block;
}

.badge-success {
  color: #67c23a;
  font-weight: bold;
  margin-left: 10px;
}

.badge-warning {
  color: #e6a23c;
  font-weight: bold;
  margin-left: 10px;
}

.badge-error {
  color: #f56c6c;
  font-weight: bold;
  margin-left: 10px;
}

.preview-note {
  color: #909399;
  font-size: 13px;
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px dashed #dcdfe6;
}

.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
}

/* Element Plus 组件样式微调 */
:deep(.el-input-number) {
  width: 200px;
}

:deep(.el-input__inner) {
  text-align: left;
}

:deep(.el-alert__title) {
  font-size: 14px;
  margin-bottom: 10px;
}

/* 响应式设计 */
@media screen and (max-width: 768px) {
  .form-item-full {
    width: 100%;
  }
  
  .form-item-full :deep(.el-form-item__content) {
    flex-direction: column;
    align-items: flex-start;
  }
  
  .param-tip {
    margin-left: 0;
    margin-top: 5px;
  }
}
</style>
