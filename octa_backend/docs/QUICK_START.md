# 🚀 OCTA血管分割平台 - 快速启动指南

**版本：** v1.0.0 | **更新：** 2026-01-28 | **耗时：** 5分钟

---

## ⚡ 5分钟快速开始

### 前置条件 ✅
- Python 3.8+（已安装）
- FastAPI + PyTorch（已在虚拟环境中）
- Vue 3 + Vite（前端依赖已装）

### 步骤1：启动后端（必须先启动）

```bash
# 进入后端目录
cd octa_backend

# 启动服务（自动化脚本）
python main.py

# 或使用 uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**预期输出：**
```
🚀 服务启动参数
========================================================================
  监听地址: 0.0.0.0:8000
  访问地址: http://127.0.0.1:8000 或 http://localhost:8000
  API文档: http://127.0.0.1:8000/docs (Swagger UI)
========================================================================
```

✅ **后端成功启动！**

---

### 步骤2：启动前端（新开终端）

```bash
# 进入前端目录
cd octa_frontend

# 首次启动需要装依赖
npm install

# 启动开发服务器
npm run dev
```

**预期输出：**
```
VITE v4.x.x  ready in 123 ms

  ➜  Local:   http://127.0.0.1:5173/
  ➜  press h + enter to show help
```

✅ **前端成功启动！**

---

### 步骤3：打开浏览器访问

```
http://127.0.0.1:5173/
```

或者查看后端API文档：
```
http://127.0.0.1:8000/docs
```

✅ **平台启动完成！** 🎉

---

## 🧪 验证功能是否正常

### 方式1：使用前端UI（推荐）

1. 打开 http://127.0.0.1:5173/
2. 选择一张OCTA图像（PNG格式）
3. 点击"上传"按钮
4. 等待分割结果显示

### 方式2：运行自动化测试脚本

```bash
cd octa_backend

# 运行完整测试（包含8个测试步骤）
python test_seg_api.py
```

预期结果：
```
✓ 测试1：后端健康检查 - PASS
✓ 测试2：权重管理接口 - PASS
✓ 测试3：图像分割接口 - PASS
✓ 测试4：模型选择功能 - PASS
... (共8个测试)
✓ 所有测试通过！
```

---

## 📖 常用命令速查

| 操作 | 命令 |
|-----|------|
| 启动后端 | `cd octa_backend && python main.py` |
| 启动前端 | `cd octa_frontend && npm run dev` |
| 运行测试 | `python test_seg_api.py` |
| 查看API | `http://127.0.0.1:8000/docs` |
| 访问平台 | `http://127.0.0.1:5173/` |
| 检查环境 | `python check_backend.py` |

---

## 🎯 下一步

- ✅ 已启动平台 → 开始上传图像进行分割
- 📖 想深入了解 → 查看 [文档导航中心](../DOCUMENTATION_INDEX.md)
- 🧪 想进行完整测试 → 查看 [联调测试指南](./INTEGRATION_TEST_GUIDE.md)
- 🔧 想修改配置参数 → 查看 [配置使用指南](./CONFIG_USAGE_GUIDE.md)
- 🐛 遇到问题 → 查看 [故障排查手册](./TROUBLESHOOTING.md)

---

## ⚠️ 常见问题速解

### Q: 跨域报错？
**A:** 确保前端运行在 `http://127.0.0.1:5173/`（不要用localhost）

### Q: 后端启动失败？
**A:** 检查虚拟环境是否激活，依赖是否完整
```bash
# 重新安装依赖
pip install -r requirements.txt
```

### Q: 前端无法连接后端？
**A:** 确保后端已启动且运行在 `http://127.0.0.1:8000`

### Q: 端口被占用？
**A:** 修改启动命令的端口号
```bash
python main.py  # 更改端口配置
```

---

## 🎓 获取帮助

| 需求 | 查看文档 |
|-----|--------|
| 快速上手 | 本文件 ✓ |
| 完整测试 | [联调测试指南](./INTEGRATION_TEST_GUIDE.md) |
| 参数配置 | [配置使用指南](./CONFIG_USAGE_GUIDE.md) |
| 故障排查 | [故障排查手册](./TROUBLESHOOTING.md) |
| 项目架构 | [项目结构说明](./PROJECT_STRUCTURE.md) |
| 所有文档 | [文档导航中心](../DOCUMENTATION_INDEX.md) |

---

**祝您使用愉快！** 🎉

如有问题，参考 [文档导航中心](../DOCUMENTATION_INDEX.md) 或查看 [故障排查手册](./TROUBLESHOOTING.md)
