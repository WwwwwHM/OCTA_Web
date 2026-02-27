# 测试数据目录

## 📁 目录说明

此目录用于存放联调测试所需的测试图像。

---

## 📋 必需文件

### test_image.png

**用途：** 联调测试的输入图像

**要求：**
- ✅ **格式：** PNG（推荐）/ JPG / BMP / TIFF
- ✅ **类型：** 灰度图像（单通道）
- ✅ **尺寸：** 256x256 像素（推荐）
- ✅ **内容：** OCTA血管成像图像

**获取方式：**

1. **使用项目自带的测试图像**（如果有）
   ```bash
   # 从其他目录复制
   copy ..\uploads\sample.png test_image.png
   ```

2. **使用您自己的OCTA图像**
   ```bash
   # 将您的图像重命名并放入此目录
   copy C:\path\to\your\octa_image.png test_image.png
   ```

3. **从数据集下载**
   - 公开OCTA数据集（如ROSE、OCTA-500等）
   - 选择一张血管清晰的图像
   - 预处理为256x256灰度图

---

## 📝 创建测试图像

### 方式1：使用Python脚本生成

```python
# generate_test_image.py
from PIL import Image
import numpy as np

# 创建256x256灰度图（示例）
img_array = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
img = Image.fromarray(img_array, mode='L')
img.save('test_data/test_image.png')
print("测试图像已生成")
```

### 方式2：从现有图像转换

```python
from PIL import Image

# 读取现有图像
img = Image.open('your_original_image.jpg')

# 转换为灰度
img_gray = img.convert('L')

# 调整尺寸
img_resized = img_gray.resize((256, 256))

# 保存
img_resized.save('test_data/test_image.png')
```

---

## ✅ 验证测试图像

运行以下命令验证图像格式正确：

```python
from PIL import Image
import numpy as np

# 加载图像
img = Image.open('test_data/test_image.png')

# 验证信息
print(f"图像尺寸: {img.size}")           # 预期: (256, 256)
print(f"图像模式: {img.mode}")           # 预期: L（灰度）
print(f"像素值范围: {np.array(img).min()}-{np.array(img).max()}")  # 预期: 0-255
```

**预期输出：**
```
图像尺寸: (256, 256)
图像模式: L
像素值范围: 0-255
```

---

## 🎯 测试图像示例

### 理想的测试图像特征

✅ **血管结构清晰**  
✅ **对比度良好**（血管与背景区分明显）  
✅ **无明显伪影**  
✅ **尺寸标准**（256x256）  
✅ **格式规范**（8位灰度PNG）

### 不合适的测试图像

❌ RGB彩色图像（需转为灰度）  
❌ 尺寸过大或过小（需resize到256x256）  
❌ 过度模糊或噪声严重  
❌ 非医学影像图像

---

## 📂 目录结构

```
test_data/
├── test_image.png          # 主测试图像（必需）
├── test_image_2.png        # 备用测试图像（可选）
├── test_image_3.png        # 备用测试图像（可选）
└── README.md               # 本说明文档
```

---

## 🔗 相关资源

### OCTA公开数据集

1. **ROSE Dataset**
   - 网址: http://rose.ahu.edu.cn/
   - 包含: 多种OCTA病变图像

2. **OCTA-500 Dataset**
   - 网址: https://ieee-dataport.org/open-access/octa-500
   - 包含: 500眼OCTA图像

3. **其他数据集**
   - 联系医学影像研究机构
   - 使用实验室已有数据

---

## ⚠️ 注意事项

1. **版权和隐私**
   - 确保有权使用测试图像
   - 不要使用真实患者数据（除非已脱敏）

2. **数据安全**
   - 不要将敏感医学图像上传到公开仓库
   - 建议使用合成数据或公开数据集

3. **测试覆盖**
   - 准备多样化的测试图像（正常、病变、不同质量）
   - 验证模型在各种情况下的表现

---

**最后更新：** 2026-01-28  
**维护者：** OCTA Web项目组
