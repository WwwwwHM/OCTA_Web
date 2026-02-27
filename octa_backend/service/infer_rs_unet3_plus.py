"""
RS-Unet3+ 推理与评估服务

功能概览：
- infer_single_image: 单张 OCTA 图像推理，输出掩码、FAZ 区域坐标与面积
- evaluate_dataset: 批量评估（血管 / FAZ 的 Dice、IoU、Precision、Recall、FAZ 面积误差、血管迂曲度）
- 结果保存：掩码 PNG、评估 JSON 报告、叠加可视化、指标曲线

实现要点：
- 预处理：Resize -> 归一化 [0,1] -> CHW -> Batch
- 后处理：阈值 0.5 -> 去除小连通域
- 设备：GPU 优先，fallback CPU
- 断点继续：直接加载权重进行推理
"""

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from config.config import MODEL_DIR, RESULT_DIR
from models.rs_unet3_plus import RSUNet3Plus

# 目标推理尺寸（按需求使用 512x512）
INFER_SIZE = (512, 512)


# ==================== 基础工具 ====================

def _init_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model(weight_path: str, device: torch.device) -> RSUNet3Plus:
    model = RSUNet3Plus(in_channels=3, num_classes=1).to(device)
    if weight_path and os.path.exists(weight_path):
        state = torch.load(weight_path, map_location=device)
        if isinstance(state, dict) and "model_state" in state:
            model.load_state_dict(state["model_state"])
        else:
            model.load_state_dict(state)
    model.eval()
    return model


def _preprocess_image(image_path: str) -> Tuple[torch.Tensor, Tuple[int, int]]:
    image = Image.open(image_path).convert("RGB")
    orig_size = image.size  # (w, h)
    image = TF.resize(image, INFER_SIZE)
    tensor = TF.to_tensor(image)
    tensor = TF.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return tensor.unsqueeze(0), orig_size


def _connected_components_filter(mask: np.ndarray, min_size: int = 50) -> np.ndarray:
    """简单连通域过滤，去除小伪影。"""
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    output = np.zeros_like(mask, dtype=np.uint8)
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for i in range(h):
        for j in range(w):
            if mask[i, j] == 0 or visited[i, j]:
                continue
            stack = [(i, j)]
            coords = []
            visited[i, j] = True
            while stack:
                x, y = stack.pop()
                coords.append((x, y))
                for dx, dy in dirs:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] == 1 and not visited[nx, ny]:
                        visited[nx, ny] = True
                        stack.append((nx, ny))
            if len(coords) >= min_size:
                for (x, y) in coords:
                    output[x, y] = 1
    return output


def _postprocess_mask(prob: torch.Tensor, orig_size: Tuple[int, int]) -> np.ndarray:
    prob_np = prob.squeeze().cpu().numpy()
    mask = (prob_np > 0.5).astype(np.uint8)
    mask = _connected_components_filter(mask, min_size=50)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    mask_img = mask_img.resize(orig_size, Image.Resampling.NEAREST)
    return np.array(mask_img) // 255


def _faz_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _faz_area(mask: np.ndarray) -> int:
    return int(mask.sum())


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen 细化的简化实现。"""
    skel = mask.copy().astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for iter_idx in range(2):
            to_remove = []
            for i in range(1, skel.shape[0] - 1):
                for j in range(1, skel.shape[1] - 1):
                    if skel[i, j] == 0:
                        continue
                    p2 = skel[i - 1, j]
                    p3 = skel[i - 1, j + 1]
                    p4 = skel[i, j + 1]
                    p5 = skel[i + 1, j + 1]
                    p6 = skel[i + 1, j]
                    p7 = skel[i + 1, j - 1]
                    p8 = skel[i, j - 1]
                    p9 = skel[i - 1, j - 1]
                    neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
                    transitions = sum((neighbors[k] == 0 and neighbors[(k + 1) % 8] == 1) for k in range(8))
                    nz = sum(neighbors)
                    if 2 <= nz <= 6 and transitions == 1:
                        if iter_idx == 0:
                            if p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0:
                                to_remove.append((i, j))
                        else:
                            if p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0:
                                to_remove.append((i, j))
            if to_remove:
                changed = True
                for x, y in to_remove:
                    skel[x, y] = 0
    return skel


def _vessel_tortuosity(mask: np.ndarray) -> float:
    skel = _skeletonize(mask)
    pts = np.argwhere(skel > 0)
    if pts.shape[0] < 2:
        return 0.0
    length = float(pts.shape[0])
    # 近似最远两点直线距离
    dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    max_dist = dists.max() + 1e-6
    return float(length / max_dist)


def _precision_recall(pred: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    tp = np.logical_and(pred == 1, target == 1).sum()
    fp = np.logical_and(pred == 1, target == 0).sum()
    fn = np.logical_and(pred == 0, target == 1).sum()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return float(precision), float(recall)


def _dice_iou(pred: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    intersection = np.logical_and(pred == 1, target == 1).sum()
    dice = (2 * intersection) / (pred.sum() + target.sum() + 1e-6)
    union = pred.sum() + target.sum() - intersection
    iou = intersection / (union + 1e-6)
    return float(dice), float(iou)


# ==================== 推理单张 ====================

def infer_single_image(image_path: str, weight_path: str) -> Dict:
    """单张 OCTA 推理，返回掩码路径、FAZ 框与面积。"""
    device = _init_device()
    model = _load_model(weight_path, device)

    tensor, orig_size = _preprocess_image(image_path)
    tensor = tensor.to(device)

    start = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000

    mask = _postprocess_mask(probs, orig_size)
    faz_box = _faz_bbox(mask)
    faz_area = _faz_area(mask)

    # 保存掩码
    os.makedirs(RESULT_DIR, exist_ok=True)
    out_name = os.path.splitext(os.path.basename(image_path))[0] + "_seg.png"
    out_path = os.path.join(RESULT_DIR, out_name)
    Image.fromarray((mask * 255).astype(np.uint8)).save(out_path)

    _save_overlay(image_path, mask, out_path.replace("_seg.png", "_overlay.png"), faz_box)

    return {
        "mask_path": out_path,
        "faz_bbox": faz_box,
        "faz_area": faz_area,
        "latency_ms": latency_ms,
    }


# ==================== 数据集评估 ====================

@dataclass
class EvalReport:
    vessel_dice: float
    vessel_iou: float
    vessel_precision: float
    vessel_recall: float
    faz_dice: float
    faz_iou: float
    faz_precision: float
    faz_recall: float
    faz_area_abs_err: float
    faz_area_rel_err: float
    vessel_tortuosity: float
    avg_latency_ms: float
    samples: int


def _load_pair(image_path: str, mask_path: str) -> Tuple[np.ndarray, np.ndarray]:
    img = Image.open(image_path).convert("L")
    m = Image.open(mask_path).convert("L")
    img = np.array(img)
    m = (np.array(m) > 127).astype(np.uint8)
    return img, m


def evaluate_dataset(dataset_path: str, weight_path: str) -> EvalReport:
    device = _init_device()
    model = _load_model(weight_path, device)

    img_dir = os.path.join(dataset_path, "images")
    vessel_dir = os.path.join(dataset_path, "masks")
    faz_dir = os.path.join(dataset_path, "faz_masks")

    names = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(("png", "jpg", "jpeg"))])

    vessel_dice = []
    vessel_iou = []
    vessel_prec = []
    vessel_rec = []
    faz_dice = []
    faz_iou = []
    faz_prec = []
    faz_rec = []
    faz_abs_err = []
    faz_rel_err = []
    tort_list = []
    latencies = []

    for name in names:
        stem = os.path.splitext(name)[0]
        image_path = os.path.join(img_dir, name)
        vessel_gt_path = os.path.join(vessel_dir, f"{stem}.png")
        faz_gt_path = os.path.join(faz_dir, f"{stem}.png")

        tensor, orig_size = _preprocess_image(image_path)
        tensor = tensor.to(device)
        start = time.perf_counter()
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.sigmoid(logits)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

        pred_mask = _postprocess_mask(probs, orig_size)

        _, vessel_gt = _load_pair(image_path, vessel_gt_path)
        _, faz_gt = _load_pair(image_path, faz_gt_path)

        v_dice, v_iou = _dice_iou(pred_mask, vessel_gt)
        v_p, v_r = _precision_recall(pred_mask, vessel_gt)
        f_dice, f_iou = _dice_iou(pred_mask, faz_gt)
        f_p, f_r = _precision_recall(pred_mask, faz_gt)

        faz_pred_area = pred_mask.sum()
        faz_gt_area = faz_gt.sum() + 1e-6
        faz_abs_err.append(abs(faz_pred_area - faz_gt_area))
        faz_rel_err.append(abs(faz_pred_area - faz_gt_area) / faz_gt_area)

        vessel_dice.append(v_dice)
        vessel_iou.append(v_iou)
        vessel_prec.append(v_p)
        vessel_rec.append(v_r)
        faz_dice.append(f_dice)
        faz_iou.append(f_iou)
        faz_prec.append(f_p)
        faz_rec.append(f_r)

        tort_list.append(_vessel_tortuosity(pred_mask))

        # 保存叠加图与掩码
        out_mask = os.path.join(RESULT_DIR, f"{stem}_seg.png")
        Image.fromarray((pred_mask * 255).astype(np.uint8)).save(out_mask)
        _save_overlay(image_path, pred_mask, out_mask.replace("_seg.png", "_overlay.png"), _faz_bbox(pred_mask))

    report = EvalReport(
        vessel_dice=float(np.mean(vessel_dice)),
        vessel_iou=float(np.mean(vessel_iou)),
        vessel_precision=float(np.mean(vessel_prec)),
        vessel_recall=float(np.mean(vessel_rec)),
        faz_dice=float(np.mean(faz_dice)),
        faz_iou=float(np.mean(faz_iou)),
        faz_precision=float(np.mean(faz_prec)),
        faz_recall=float(np.mean(faz_rec)),
        faz_area_abs_err=float(np.mean(faz_abs_err)),
        faz_area_rel_err=float(np.mean(faz_rel_err)),
        vessel_tortuosity=float(np.mean(tort_list)),
        avg_latency_ms=float(np.mean(latencies)),
        samples=len(names),
    )

    os.makedirs(RESULT_DIR, exist_ok=True)
    json_path = os.path.join(RESULT_DIR, "rs_unet3p_eval_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report.__dict__, f, ensure_ascii=False, indent=2)

    _plot_dice_curves(vessel_dice, faz_dice, os.path.join(RESULT_DIR, "rs_unet3p_dice_curve.png"))

    return report


# ==================== 可视化 ====================

def _save_overlay(image_path: str, mask: np.ndarray, save_path: str, bbox: Optional[Tuple[int, int, int, int]]):
    image = Image.open(image_path).convert("RGB")
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img = mask_img.convert("RGBA")
    r, g, b, a = mask_img.split()
    a = a.point(lambda p: 80)
    mask_colored = Image.merge("RGBA", (mask_img.point(lambda p: 255), mask_img.point(lambda p: 0), mask_img.point(lambda p: 0), a))
    overlay = Image.alpha_composite(image.convert("RGBA"), mask_colored)
    if bbox:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(overlay)
        draw.rectangle(bbox, outline=(0, 255, 0, 255), width=2)
    overlay.convert("RGB").save(save_path)


def _plot_dice_curves(vessel_dice: List[float], faz_dice: List[float], save_path: str):
    plt.figure(figsize=(8, 5))
    plt.plot(vessel_dice, label="Vessel Dice", color="#1677ff")
    plt.plot(faz_dice, label="FAZ Dice", color="#ff4d4f")
    plt.xlabel("Sample Index")
    plt.ylabel("Dice")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    # 示例：需准备 images/masks/faz_masks 结构
    sample_dataset = os.path.join("./uploads", "sample_train")
    weight = os.path.join(MODEL_DIR, "weights", "rs_unet3p_best.pth")
    rep = evaluate_dataset(sample_dataset, weight)
    print(rep)
