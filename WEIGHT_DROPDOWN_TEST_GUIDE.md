# 权重下拉框修复 - 快速测试指南

## 🚀 快速验证清单（5分钟）

### 步骤1: 启动服务
```bash
# 终端1: 启动后端（激活虚拟环境）
cd octa_backend
..\octa_env\Scripts\activate
python main.py

# 终端2: 启动前端
cd octa_frontend
npm run dev
```

### 步骤2: 打开前端
- 访问 `http://127.0.0.1:5173`
- 打开浏览器 DevTools (`F12` -> `Console` 标签)

### 步骤3: 测试权重加载

#### 测试3.1: U-Net 权重
```
动作：
1. 刷新页面
2. 选择 "U-Net（推荐）"
3. 观察权重下拉框

预期结果：
✓ Console 显示: [权重获取] ✓ 成功加载 X 个unet权重
✓ 权重下拉框显示可用权重列表
✓ 可点击选择权重
```

#### 测试3.2: RS-Unet3+ 权重（无权重时）
```
动作：
1. 如果从未训练RS-Unet3+，清空 models/weights_rs_unet3_plus/ 目录
2. 刷新页面
3. 选择 "RS-Unet3+（前沿模型）"
4. 观察权重下拉框

预期结果：
✓ Console 显示: [权重获取] ✓ 成功加载 0 个RS-Unet3+权重
✓ 权重下拉框禁用（greyed out）
✓ 显示 "暂无RS-Unet3+权重文件"
✓ 消息提示 "暂无RS-Unet3+权重文件，将使用默认权重进行分割"
```

#### 测试3.3: 训练后权重立即显示
```
动作：
1. 选择 "RS-Unet3+（前沿模型）" -> 权重下拉框应禁用
2. 上传数据集并训练 RS-Unet3+ (或使用已有数据集快速训练)
   - 建议：用 samples/test_dataset 进行快速测试（~2分钟）
3. 等待训练完成
4. 刷新分割页面 (Ctrl+R 或点击刷新按钮)
5. 再次选择 RS-Unet3+

预期结果：
✓ Console 显示: [权重获取] API响应: {code: 200, ...}
✓ Console 显示: [权重获取] ✓ 成功加载 1 个RS-Unet3+权重
✓ Console 显示权重信息：
  {
    name: "rs_unet3_plus_YYYYMMDD_HHMMSS.pth",
    path: "models/weights_rs_unet3_plus/rs_unet3_plus_...",
    size: 102.4
  }
✓ 权重下拉框不再禁用
✓ 可点击选择刚训练的权重
✓ 后端 logs 显示: [INFO] 权重元信息已写入数据库: rs_unet3_plus_...
```

### 步骤4: 测试下拉框交互

#### 测试4.1: 基本交互
```
动作：
1. 点击权重下拉框 -> 应展开显示选项列表
2. 点击一个权重选项 -> 应选中该权重
3. 观察 selectedWeight 值变化

预期结果：
✓ Console 显示: [权重选择] 用户选择权重: models/weights/.../...
✓ Console 显示: [权重选择] ✓ 权重有效: rs_unet3_plus_...
✓ 权重下拉框显示选中的权重名称
```

#### 测试4.2: 清空选择
```
动作：
1. 点击权重下拉框右侧的 X 按钮 -> 清空选择

预期结果：
✓ selectedWeight 变为空字符串
✓ 权重下拉框显示 placeholder: "选择模型权重..."
✓ 分割时使用默认权重
```

#### 测试4.3: 模型切换时清空权重选择
```
动作：
1. 选择 U-Net 并选中一个权重
2. 切换到 RS-Unet3+
3. 再切换回 U-Net

预期结果：
✓ 每次切换模型后，权重选择被清空（防止跨模型使用权重）
✓ Console 显示: [模型选择] 用户切换模型: ...
✓ Console 显示: [权重获取] 开始加载...权重列表，model_type=...
```

---

## 🐛 常见问题排查

### 问题1: 权重下拉框仍然禁用（即使训练完成后）

**原因**: 权重未成功入库，或 API 返回空列表

**排查步骤**:
```javascript
// 在浏览器控制台执行：
fetch('http://127.0.0.1:8000/file/model-weights?model_type=rs_unet3_plus')
  .then(r => r.json())
  .then(d => console.log(JSON.stringify(d, null, 2)))

// 查看返回结果中 data 数组是否为空
// 如果为空，检查后端日志中是否有:
// [INFO] 权重元信息已写入数据库
```

**修复**:
1. 检查后端数据库: `sqlite3 data/octa.db "SELECT * FROM file_management WHERE file_type='weight' AND model_type='rs_unet3_plus'"`
2. 确认 `train_service.py` 中的 `FileDAO.add_file_record()` 被调用
3. 检查权重文件是否真实存在于 `models/weights_rs_unet3_plus/` 目录

### 问题2: 权重列表加载失败（提示"加载权重列表失败"）

**原因**: 网络超时或后端服务异常

**排查步骤**:
```javascript
// 检查网络连接
curl http://127.0.0.1:8000/file/model-weights?model_type=unet

// 查看 Console 中的完整错误:
// [权重获取] ✗ 错误: ... (完整错误信息)
```

**修复**:
1. 确认后端仍在运行: `curl http://127.0.0.1:8000/`
2. 检查后端防火墙/CORS 配置
3. 查看后端日志中是否有异常

### 问题3: 权重项无法点击（下拉框展开但选项不可交互）

**原因**: 权重数据缺少必要字段（`file_path` 或 `file_name`）

**排查步骤**:
```javascript
// 在控制台查看权重数据结构:
// 应该包含：file_name, file_path, file_size, model_type
// 检查 Console 中的权重验证日志:
// [权重验证] 权重项缺少必要字段: ...
```

**修复**:
1. 在后端数据库中检查权重记录:
   ```sql
   SELECT id, file_name, file_path, model_type FROM file_management 
   WHERE file_type='weight';
   ```
2. 确保所有权重记录都有 `file_name` 和 `file_path`
3. 如有缺失，手动更新或删除无效记录

### 问题4: 下拉框在加载时被禁用（应该显示"正在加载"）

**原因**: 未更新为新的 `isWeightListEmpty` 逻辑

**排查步骤**:
```javascript
// 检查是否已部署最新代码:
// npm run build
// 强制刷新浏览器: Ctrl+Shift+R 或 Cmd+Shift+R
```

**修复**:
1. 清除浏览器缓存: `Ctrl+Shift+Delete`
2. 重新 build 前端: `npm run build`
3. 重启前端: `npm run dev`

---

## 📊 性能测试

### 权重加载时间测试
```javascript
// 在 Console 中运行：
console.time('weight-fetch');
fetch('http://127.0.0.1:8000/file/model-weights?model_type=rs_unet3_plus')
  .then(r => r.json())
  .then(d => {
    console.timeEnd('weight-fetch');
    console.log('权重数量:', d.data.length);
  })

// 预期结果: weight-fetch: 50-200ms (根据数据库大小)
```

### 下拉框渲染性能
```javascript
// 权重列表超过1000个时，确保前端仍流畅
// 观察 DevTools -> Performance 标签，检查是否有卡顿
// 预期: 250ms 内完成渲染
```

---

## 📝 日志参考

### 成功的权重加载日志
```
[权重获取] 开始加载RS-Unet3+权重列表，modelType=rs_unet3_plus
[权重获取] API响应: {code: 200, msg: "找到2个rs_unet3_plus权重", data: [...]}
[权重获取] ✓ 成功加载 2 个RS-Unet3+权重
[权重获取] 权重列表详情: (2) [Object, Object]
[权重获取] 已自动选择权重: rs_unet3_plus_20260120_152030.pth
[权重获取] 加载完成
```

### 失败的权重加载日志
```
[权重获取] 开始加载RS-Unet3+权重列表，modelType=rs_unet3_plus
[权重获取] ✗ 错误: Network Error
[权重获取] 完整错误信息: Error: Network Error
加载权重列表失败: Network Error，将使用默认权重
[权重获取] 加载完成
```

### 数据验证日志
```
[权重验证] availableWeights不是数组，类型: object
[权重验证] 过滤无效权重: 5 -> 3 个
[权重验证] 权重项缺少必要字段: {file_name: "test"}
```

---

## ✅ 验证清单

在部署前，确认以下场景都能正确工作：

- [ ] 访问 HomeView，选择 U-Net，权重下拉框显示列表
- [ ] 选择 RS-Unet3+，如果已训练过，权重下拉框显示新权重
- [ ] 点击权重下拉框，可以展开选项列表
- [ ] 点击权重选项，可以选中该权重
- [ ] 切换模型时，权重选择被清空
- [ ] 训练完成后，刷新分割页面，新权重立即显示（无需手动操作）
- [ ] 权重加载失败时，显示友好的错误提示
- [ ] 浏览器 Console 中的日志清晰有序，无红色错误信息
- [ ] 下拉框交互流畅，无卡顿或延迟

---

## 🎓 学习资源

### Element Plus el-select 组件文档
https://element-plus.org/en-US/component/select.html

关键属性：
- `disabled`: 禁用状态
- `loading`: 加载中状态（显示转圈）
- `filterable`: 启用搜索
- `clearable`: 启用清空按钮
- `:key`: Vue 组件 key，强制重新渲染

### Vue 3 Computed vs Watch
- `computed`: 用于派生数据（权重列表验证）
- `watch`: 用于监听数据变化并执行副作用（加载权重列表）

### 前后端通信最佳实践
1. 前端严格验证后端数据
2. 提供详细的错误日志便于诊断
3. 实现优雅降级（失败时返回空数组而非抛出异常）
4. 显示用户友好的错误提示

---

**最后更新**: 2026-01-20  
**作者**: GitHub Copilot  
**版本**: 1.0

