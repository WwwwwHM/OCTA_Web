# 模型训练与权重管理优化 - 快速参考

## 🎯 核心改进

### 1. 训练页面优化
- **模型卡片**：U-Net（基础模型） / RS-Unet3+（高级模型）
- **视觉区分**：蓝色Info徽章 vs 绿色Success徽章
- **参数表单**：3个基础参数 vs 6个高级参数

### 2. 权重文件隔离
```
models/
├── weights_unet/               ← U-Net训练权重
│   ├── unet_20260120_143052.pth
│   └── unet_20260120_150230.pth
├── weights_rs_unet3_plus/      ← RS-Unet3+训练权重
│   ├── rs_unet3_plus_20260120_143052.pth
│   └── rs_unet3_plus_20260120_150230.pth
└── weights/                    ← 通用权重（向后兼容）
    └── unet_octa.pth
```

### 3. 分割页面联动
- 选择U-Net → 自动加载U-Net权重列表
- 选择RS-Unet3+ → 自动加载RS-Unet3+权重列表
- 切换模型 → 自动清空之前选择的权重
- 无权重 → 禁用下拉框并提示

---

## 📋 使用流程

### 训练新模型
1. 打开训练页面（/train）
2. 点击模型卡片（U-Net 或 RS-Unet3+）
3. 上传数据集ZIP压缩包
4. 设置训练参数（epochs、lr等）
5. 点击"开始训练"按钮
6. 权重自动保存到对应目录

### 使用训练权重分割
1. 打开分割页面（/）
2. 上传OCTA图像
3. 选择模型（U-Net 或 RS-Unet3+）
4. 权重下拉框自动更新
5. 选择训练权重（可选，留空使用默认）
6. 点击"开始图像分割"

---

## 🔧 配置文件位置

### 后端配置
- **权重目录**：`config/config.py` → `WEIGHT_DIR_MAP`
- **文件前缀**：`config/config.py` → `WEIGHT_PREFIX_MAP`

### 数据库表
- **表名**：`images`
- **model_type字段**：'unet' | 'rs_unet3_plus' | 'fcn'

---

## 🧪 验证测试

### 测试训练功能
```bash
# 1. 准备数据集（ZIP格式）
dataset.zip
├── images/
│   ├── img001.png
│   ├── img002.png
│   └── ...
└── masks/
    ├── img001.png
    ├── img002.png
    └── ...

# 2. 前端选择模型并上传
# 3. 检查权重保存位置
ls models/weights_unet/         # U-Net权重
ls models/weights_rs_unet3_plus/ # RS-Unet3+权重

# 4. 验证数据库记录
sqlite3 octa.db "SELECT * FROM images WHERE file_type='weight';"
```

### 测试分割功能
```bash
# 1. 打开分割页面
http://localhost:5173/

# 2. 选择U-Net模型
# 3. 验证权重下拉框仅显示U-Net权重

# 4. 选择RS-Unet3+模型
# 5. 验证权重下拉框仅显示RS-Unet3+权重
```

---

## 🔍 故障排查

### 问题1：权重列表为空
**原因**：没有对应模型的训练权重  
**解决**：先训练该模型或使用默认权重

### 问题2：权重加载失败
**原因**：model_type不匹配  
**解决**：检查数据库model_type字段是否正确

### 问题3：训练后权重未出现
**原因**：权重保存路径错误  
**解决**：检查config.py中的WEIGHT_DIR_MAP配置

---

## 📊 关键API端点

### 训练接口
```
POST /train/upload-dataset
参数：
  - file: ZIP压缩包
  - model_arch: 'unet' | 'rs_unet3_plus'
  - epochs: 训练轮数
  - lr: 学习率
  - batch_size: 批次大小
```

### 权重查询接口
```
GET /file/model-weights?model_type={model_type}
参数：
  - model_type: 'unet' | 'rs_unet3_plus'
响应：
  - data: 权重文件列表（已按模型类型筛选）
```

---

## ✨ 新增功能

### 模型徽章
- **基础模型**（U-Net）：蓝色Info标签
- **高级模型**（RS-Unet3+）：绿色Success标签

### 自动权重加载
```javascript
// HomeView.vue 监听器
watch(selectedModel, async (newModel) => {
  selectedWeight.value = ''  // 清空之前选择
  await fetchWeights(newModel)  // 加载新模型权重
})
```

### 智能提示
- 无权重时自动提示："暂无XXX模型的权重文件，将使用默认权重"
- 权重下拉框自动禁用（无权重时）

---

## 🎉 完成状态

✅ **训练页面**：模型徽章、参数表单条件渲染  
✅ **权重隔离**：按模型类型分目录存储  
✅ **分割联动**：自动加载对应权重列表  
✅ **数据库约束**：model_type字段验证  
✅ **前端编译**：0错误，所有功能正常  
✅ **开发服务器**：运行中（http://localhost:5173）

---

**文档版本**：v1.0  
**更新日期**：2026年1月20日  
**状态**：✅ 实施完成，已验证
