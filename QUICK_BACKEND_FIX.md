# 🔧 前端训练失败故障排查（1分钟版）

## 🎯 症状
```
错误: ERR_CONNECTION_RESET / Network Error
地址: POST http://127.0.0.1:8000/train/upload-dataset
```

---

## ⚡ 一行命令检查后端

**在新终端运行：**
```bash
curl http://127.0.0.1:8000/
```

### 如果显示 JSON 响应 ✓
→ 后端正常，问题可能在前端请求

### 如果显示 "拒绝连接" ✗
→ 后端没启动，继续下面步骤

---

## 🚀 快速启动后端（选一个）

### 方法A：最简单（推荐）
```bash
cd octa_backend
python main.py
```

### 方法B：带诊断
```bash
cd octa_backend
python quick_diagnose.py
```

### 方法C：确保依赖
```bash
cd octa_backend
pip install -r requirements.txt
python main.py
```

---

## ✅ 成功标志

控制台应显示：
```
[INFO] Service address: 127.0.0.1:8000
[SUCCESS] File management table is ready
```

然后访问 http://127.0.0.1:5173 重新点击训练

---

## 🔍 如果后端启动失败

**查看错误信息，最常见的3个问题：**

### 问题1️⃣：`ModuleNotFoundError: No module named 'albumentations'`
```bash
pip install albumentations>=1.3.0
python main.py
```

### 问题2️⃣：`ModuleNotFoundError: No module named 'models.xxx'`
```bash
pip install -r requirements.txt
python main.py
```

### 问题3️⃣：`Address already in use: 127.0.0.1:8000`
另一个后端实例已在运行，关闭它或指定不同端口：
```bash
uvicorn main:app --host 127.0.0.1 --port 8001
```

---

## 📊 完整诊断步骤

```
1. 打开新终端
   ↓
2. cd octa_backend
   ↓
3. ..\octa_env\Scripts\activate     (激活虚拟环境)
   ↓
4. pip install -r requirements.txt  (确保依赖)
   ↓
5. python main.py                   (启动后端)
   ↓
6. 看到 [SUCCESS] 日志 → 成功 ✓
   ↓
7. 回到前端，重新点击训练
```

---

**该步骤通常能解决99%的连接问题！**

