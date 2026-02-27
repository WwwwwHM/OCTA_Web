/**
 * OCTA全局状态管理
 * 使用Vue3 Composition API实现简单的全局状态管理
 * 功能：管理全局模型架构选择（unet/rs_unet3_plus）
 */

import { ref, readonly } from 'vue'

// ==================== 全局状态 ====================

/**
 * 全局选中的模型架构
 * - 'unet': U-Net通用模型
 * - 'rs_unet3_plus': RS-Unet3+专用模型（OCTA血管/FAZ分割）
 * - 'fcn': FCN模型
 */
const globalModelArch = ref('unet')

/**
 * RS-Unet3+模型是否可用
 * 根据后端接口或本地检测确定
 */
const rsUnet3PlusAvailable = ref(true)

// ==================== 状态操作方法 ====================

/**
 * 设置全局模型架构
 * @param {string} arch - 模型架构类型 ('unet' | 'rs_unet3_plus' | 'fcn')
 */
export function setGlobalModelArch(arch) {
  if (!['unet', 'rs_unet3_plus', 'fcn'].includes(arch)) {
    console.warn(`无效的模型架构类型: ${arch}，默认使用 unet`)
    arch = 'unet'
  }
  globalModelArch.value = arch
  console.log(`全局模型架构已切换为: ${arch}`)
}

/**
 * 获取全局模型架构（只读）
 * @returns {string} 当前全局模型架构
 */
export function getGlobalModelArch() {
  return globalModelArch.value
}

/**
 * 设置RS-Unet3+模型可用性
 * @param {boolean} available - 是否可用
 */
export function setRsUnet3PlusAvailable(available) {
  rsUnet3PlusAvailable.value = available
}

/**
 * 获取RS-Unet3+模型可用性（只读）
 * @returns {boolean} RS-Unet3+是否可用
 */
export function getRsUnet3PlusAvailable() {
  return readonly(rsUnet3PlusAvailable)
}

/**
 * 获取模型显示名称
 * @param {string} arch - 模型架构类型
 * @returns {string} 模型显示名称
 */
export function getModelDisplayName(arch) {
  const names = {
    'unet': 'U-Net',
    'rs_unet3_plus': 'RS-Unet3+',
    'fcn': 'FCN'
  }
  return names[arch] || arch
}

// ==================== 导出全局状态（响应式引用）====================

export function useGlobalState() {
  return {
    // 响应式状态
    globalModelArch: readonly(globalModelArch),
    rsUnet3PlusAvailable: readonly(rsUnet3PlusAvailable),
    
    // 操作方法
    setGlobalModelArch,
    getGlobalModelArch,
    setRsUnet3PlusAvailable,
    getRsUnet3PlusAvailable,
    getModelDisplayName
  }
}
