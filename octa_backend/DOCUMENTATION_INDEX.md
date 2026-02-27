# 📚 OCTA血管分割平台 - 文档导航中心

**最后更新：** 2026-01-28 | **版本：** v1.0.0 | **状态：** ✅ 已整理和优化

---

## 🎯 按场景快速选择文档

### 👤 我是首次使用，想快速开始（5分钟）

**→ 看这个：** [`docs/QUICK_START.md`](./docs/QUICK_START.md) ⭐ **新手必读**

```
包含内容：
├─ 1️⃣ 前置条件检查
├─ 2️⃣ 后端启动（3行代码）
├─ 3️⃣ 前端启动（3行代码）
├─ 4️⃣ 浏览器访问
└─ 5️⃣ 验证功能

预计时间：5分钟
难度：⭐☆☆☆☆
```

---

### 📊 我想进行完整的联调测试（验证所有功能）

**→ 看这个：** [`docs/INTEGRATION_TEST_GUIDE.md`](./docs/INTEGRATION_TEST_GUIDE.md) ⭐⭐ **重要**

```
包含内容：
├─ 环境准备
├─ 8步骤完整测试流程
├─ 自动化测试脚本说明
├─ 期望结果和日志
├─ 答辩演示建议
└─ 故障排查

预计时间：30分钟
难度：⭐⭐☆☆☆
适合：答辩演示、完整验证
```

---

### 🔧 我想修改配置参数（模型、端口、日志等）

**→ 看这个：** [`docs/CONFIG_USAGE_GUIDE.md`](./docs/CONFIG_USAGE_GUIDE.md)

```
包含内容：
├─ 所有配置参数详解（7大类）
├─ 参数修改示例
├─ 配置的实时生效说明
└─ 推荐配置方案

预计时间：15分钟
难度：⭐⭐☆☆☆
```

---

### 🐛 我遇到问题，需要排查（端口占用、跨域错误等）

**→ 看这个：** [`docs/TROUBLESHOOTING.md`](./docs/TROUBLESHOOTING.md)

```
包含内容：
├─ 常见错误及解决方案
├─ 日志读取方法
├─ 环境验证步骤
└─ 联系方式

预计时间：10分钟
难度：⭐☆☆☆☆
```

---

### 📋 我要验证我的测试（使用检查清单记录）

**→ 看这个：** [`docs/TEST_CHECKLIST.md`](./docs/TEST_CHECKLIST.md)

```
包含内容：
├─ 逐项测试清单
├─ 预期结果记录表
├─ 测试通过/失败判定
└─ 问题记录区

用途：在进行联调测试时使用，记录每个步骤的结果
```

---

### 🏗️ 我想了解项目架构和代码结构

**→ 看这个：** [`docs/PROJECT_STRUCTURE.md`](./docs/PROJECT_STRUCTURE.md)

```
包含内容：
├─ 目录树结构
├─ 各模块职责说明
├─ 数据流向图
├─ 关键文件位置
└─ 扩展开发指南

预计时间：15分钟
难度：⭐⭐☆☆☆
```

---

### 💾 我想查看数据库相关信息（表结构、SQL等）

**→ 看这个：** [`docs/SQL_REFERENCE.md`](./docs/SQL_REFERENCE.md)

```
包含内容：
├─ 数据库设计
├─ 表结构说明
├─ 常用SQL查询
└─ 数据操作示例
```

---

### 📦 我想了解本次生成的测试包

**→ 看这个：** [`README_TEST_PACKAGE.md`](./README_TEST_PACKAGE.md)

```
包含内容：
├─ 测试文件清单
├─ 各文件的用途
├─ 使用流程
└─ 集成说明
```

---

### 📖 我想要所有文档的快速参考卡片

**→ 看这个：** [`QUICK_REFERENCE.txt`](./QUICK_REFERENCE.txt)

```
包含内容：
├─ 常用命令速查表
├─ API端点速查
├─ 快速链接导航
└─ 应急命令

用途：贴在桌子上，随时速查
```

---

### 🎓 我要写毕设论文/学位论文的系统章节

**→ 看这个：** [`docs/THESIS_WRITING_GUIDE.md`](./docs/THESIS_WRITING_GUIDE.md) ⭐⭐⭐ **论文必读**

```
包含内容：
├─ 论文章节规划（章节分配、字数建议）
├─ 各章节详细写作要求
├─ 技术内容呈现规范
├─ 图表和代码写作规范
├─ 学术语言要求
├─ OCTA项目特定指导
├─ 常见问题和改进建议
└─ 论文写作检查清单

预计时间：根据章节长度而定
难度：⭐⭐⭐（需要深度理解项目）
适合：所有需要撰写毕设论文的同学
```

---

## 📂 文档体系结构

```
OCTA_Web/
│
├─ 📖 快速入门（首先看这些）
│  ├─ docs/QUICK_START.md          ⭐ 新手5分钟快速开始
│  ├─ QUICK_REFERENCE.txt          速查命令卡片
│  └─ DOCUMENTATION_INDEX.md        文档导航（本文件）
│
├─ 🧪 联调测试和验证（答辩前必读）
│  ├─ docs/INTEGRATION_TEST_GUIDE.md   ⭐⭐ 完整测试指南
│  ├─ docs/TEST_CHECKLIST.md          测试验证清单
│  └─ README_TEST_PACKAGE.md         测试包说明
│
├─ ⚙️ 配置和参数管理
│  ├─ docs/CONFIG_USAGE_GUIDE.md      配置参数详解
│  └─ config/config.py              配置实现文件
│
├─ 🔧 开发参考和故障排查
│  ├─ docs/PROJECT_STRUCTURE.md       项目结构说明
│  ├─ docs/TROUBLESHOOTING.md         故障排查手册
│  └─ docs/SQL_REFERENCE.md           数据库参考
│
├─ 📁 历史文档（旧版本，可参考）
│  └─ docs/_legacy/                  升级、欠拟合修复等旧文件
│
├─ 📄 根目录关键文件
│  ├─ README.md                     项目总览
│  ├─ test_seg_api.py               自动化测试脚本
│  ├─ main.py                       后端应用入口
│  └─ test_data/                    测试图像数据
│
└─ 💾 核心代码目录
   ├─ router/                      API路由
   ├─ core/                        核心模块
   ├─ config/                      配置管理
   └─ models/                      神经网络模型
```

---

## 🎓 按工作流程选择文档

### 工作流1：新用户上手（首次使用）

```
1. 阅读 QUICK_START.md        (5分钟)
   ↓
2. 启动后端和前端             (3分钟)
   ↓
3. 打开浏览器访问平台         (1分钟)
   ↓
✅ 成功！开始使用
```

---

### 工作流2：完整功能验证（答辩前必做）

```
1. 准备测试数据 (test_data/)
   ↓
2. 启动后端:     python main.py
   ↓
3. 查看测试指南: INTEGRATION_TEST_GUIDE.md
   ↓
4. 运行测试脚本: python test_seg_api.py
   ↓
5. 记录结果:     TEST_CHECKLIST.md
   ↓
✅ 所有测试通过，可以答辩！
```

---

### 工作流3：遇到问题排查（故障处理）

```
1. 看错误信息
   ↓
2. 搜索 TROUBLESHOOTING.md
   ↓
3. 按建议操作
   ↓
4. 运行 test_seg_api.py 验证
   ↓
✅ 问题解决
```

---

### 工作流4：修改配置参数

```
1. 理解参数含义: CONFIG_USAGE_GUIDE.md
   ↓
2. 修改文件:     config/config.py
   ↓
3. 重启服务:     python main.py
   ↓
4. 验证效果:     python test_seg_api.py
   ↓
✅ 配置生效
```

---

## 🎯 文档优先级

### ⭐⭐⭐ 必读（新手必须）
- `docs/QUICK_START.md` - 快速上手
- `docs/INTEGRATION_TEST_GUIDE.md` - 完整测试

### ⭐⭐ 重要（经常查阅）
- `docs/CONFIG_USAGE_GUIDE.md` - 参数配置
- `docs/TROUBLESHOOTING.md` - 故障排查
- `docs/TEST_CHECKLIST.md` - 测试验证

### ⭐ 参考（按需查看）
- `docs/PROJECT_STRUCTURE.md` - 项目结构
- `docs/SQL_REFERENCE.md` - 数据库
- `test_data/README.md` - 测试数据
- `README_TEST_PACKAGE.md` - 测试包说明

---

## ❓ 常见问题速查

| 问题 | 答案 | 查看文档 |
|-----|------|--------|
| 怎么快速启动? | 3行代码启动后端+前端 | [`QUICK_START.md`](./docs/QUICK_START.md) |
| 怎么进行完整测试? | 按8步骤进行自动化测试 | [`INTEGRATION_TEST_GUIDE.md`](./docs/INTEGRATION_TEST_GUIDE.md) |
| 跨域报错怎么办? | 检查前端端口配置 | [`TROUBLESHOOTING.md`](./docs/TROUBLESHOOTING.md) |
| 怎么修改配置? | 修改config.py中的参数 | [`CONFIG_USAGE_GUIDE.md`](./docs/CONFIG_USAGE_GUIDE.md) |
| 项目结构是怎样的? | 查看目录树和模块说明 | [`PROJECT_STRUCTURE.md`](./docs/PROJECT_STRUCTURE.md) |
| 数据库表结构? | 查看DAO和SQL文档 | [`SQL_REFERENCE.md`](./docs/SQL_REFERENCE.md) |

---

## 🔗 快速链接

### 启动命令
```bash
# 启动后端
cd octa_backend && python main.py

# 启动前端（新终端）
cd octa_frontend && npm run dev

# 运行测试
cd octa_backend && python test_seg_api.py
```

### 访问地址
```
后端服务:  http://127.0.0.1:8000
前端应用:  http://127.0.0.1:5173
API文档:   http://127.0.0.1:8000/docs
```

### 主要文件
```
config/config.py         ← 修改这里来配置参数
main.py                  ← 后端应用入口
test_seg_api.py          ← 自动化测试脚本
test_data/              ← 测试图像目录
```

---

## ✨ 文档更新日志

### v1.0.0 (2026-01-28) - 文档体系优化 ✅

✅ **文档整理成果：**
- 🗑️ 删除重复文件：QUICK_START.md (4个变体)、README_UPDATE_SUMMARY.md
- 📁 整理核心文档到 `docs/` 目录（6个文件）
- 📁 归档旧文件到 `docs/_legacy/`（6个升级/修复阶段文件）
- 📝 创建统一的 QUICK_START.md（新手友好）
- 🎯 重写 DOCUMENTATION_INDEX.md（清晰导航）

✅ **当前文档体系：**
- 📂 总文件数：从46个减少到14个核心 + 6个归档
- 📖 文档清晰度：++++ 从混乱变得结构清晰
- 🎯 新手友好度：++++ 按场景明确指引

---

## 🚀 下一步

1. **立即开始** → 阅读 [`docs/QUICK_START.md`](./docs/QUICK_START.md)
2. **进行测试** → 查看 [`docs/INTEGRATION_TEST_GUIDE.md`](./docs/INTEGRATION_TEST_GUIDE.md)
3. **遇到问题** → 参考 [`docs/TROUBLESHOOTING.md`](./docs/TROUBLESHOOTING.md)
4. **修改配置** → 查看 [`docs/CONFIG_USAGE_GUIDE.md`](./docs/CONFIG_USAGE_GUIDE.md)
5. **了解架构** → 查看 [`docs/PROJECT_STRUCTURE.md`](./docs/PROJECT_STRUCTURE.md)

---

**📍 文档位置：** `OCTA_Web/octa_backend/DOCUMENTATION_INDEX.md`  
**📧 维护者：** OCTA Web项目组  
**✅ 状态：** 已整理优化，推荐使用  
**💾 历史文档：** `docs/_legacy/` 目录（可参考，不影响使用）

---

**🎉 文档导航优化完成！享受清晰的开发体验！**

