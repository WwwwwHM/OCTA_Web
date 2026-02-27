# GitHub 发布前检查清单（OCTA_Web）

## 1) 仓库内容安全检查
- [ ] 确认未提交虚拟环境：`octa_env/`
- [ ] 确认未提交前端依赖：`octa_frontend/node_modules/`
- [ ] 确认未提交运行时输出：`octa_backend/uploads/`、`octa_backend/results/`
- [ ] 确认未提交敏感配置：`.env`、证书、密钥文件
- [ ] 确认未提交大模型权重：`octa_backend/models/weights/*.pth`

## 2) 医学数据与隐私
- [ ] 删除或脱敏所有可能含患者信息的图像/样本
- [ ] 检查 `test_data`、`uploads`、`results` 不包含真实隐私数据
- [ ] 若必须保留示例图片，确保为公开可分发数据集且无身份信息

## 3) 体积与 GitHub 限制
- [ ] 单文件不超过 100MB（GitHub 硬限制）
- [ ] 仓库避免提交大二进制文件（权重建议外链或下载脚本）
- [ ] 如确需跟踪大文件，使用 Git LFS

## 4) 可运行性与文档
- [ ] `README.md` 包含后端与前端启动步骤
- [ ] 明确默认端口：后端 `8000`，前端 `5173`
- [ ] 明确 CORS 端口同步修改位置：`octa_backend/main.py`
- [ ] 提供依赖安装命令（`pip install -r ...` / `npm install`）

## 5) 建议提交前命令（在仓库根目录）
```bash
git status
git check-ignore -v octa_env octa_frontend/node_modules octa_backend/uploads octa_backend/results
```

```bash
# 查看是否意外包含敏感词（可按需扩展）
git grep -nE "(API_KEY|SECRET|TOKEN|PASSWORD|PRIVATE KEY)" -- . ":(exclude).git"
```

## 6) 首次推送示例
```bash
git init
git add .
git commit -m "chore: initial publish"
git branch -M main
git remote add origin <你的仓库地址>
git push -u origin main
```

## 7) 发布后建议
- [ ] 在仓库 Settings 开启 Secret scanning / Dependabot（若可用）
- [ ] 添加 `LICENSE`（如 MIT）
- [ ] 添加 Issue/PR 模板，便于后续协作
