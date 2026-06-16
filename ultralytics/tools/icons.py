import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def _sanitize(name: str) -> str:
    return (name.replace(" ", "_")
                 .replace("+", "plus")
                 .replace("/", "_")
                 .replace("|", "_")
                 .replace(":", "_"))

def export_box_style_icons(box_list, output_dir="result_output/icons",
                           size_px=128, dpi=300, colormap_name="tab10"):
    """
    仅导出“框样式”的小图标（透明背景、无文字、无底图）。
    box_list: [(label, conf, (x1,y1,x2,y2)), ...]  # 只用到 label 顺序以保证颜色一致
    """
    os.makedirs(output_dir, exist_ok=True)
    cmap = plt.get_cmap(colormap_name)

    for idx, (label, conf, _) in enumerate(box_list, start=1):
        # 图尺寸
        fig = plt.figure(figsize=(size_px / dpi, size_px / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])  # 全画布
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # 边框参数（GT 虚线黑色；其他为 tab10 依次取色的实线）
        if str(label).strip().lower() in {"gt", "groundtruth", "ground_truth"}:
            edgecolor = "black"
            linestyle = "--"
            linewidth = 3.0
        else:
            # 预测框颜色从 0 开始：为了与 draw_boxes 一致，这里用 (idx-2)
            # 因为 idx=1 是 GT，从第二个元素开始配色索引 0,1,2,...
            color_idx = max(0, idx - 2)
            edgecolor = cmap(color_idx % 10)
            linestyle = "-"
            linewidth = 2.5

        # 在坐标系中画一个占比 0.75 的方框作为图标
        margin = 0.125
        rect = patches.Rectangle(
            (margin, margin),
            1 - 2 * margin,
            1 - 2 * margin,
            linewidth=3.5,
            edgecolor=edgecolor,
            facecolor="none",
            linestyle=linestyle
        )
        ax.add_patch(rect)

        # 透明背景保存
        safe_label = _sanitize(str(label))
        save_path = os.path.join(output_dir, f"icon_{idx:02d}_{safe_label}.png")
        fig.savefig(save_path, dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print(f"✅ 已导出图标: {save_path}")

if __name__ == "__main__":
    boxes = [
        ("GT", None, (1471.2, 1165.6, 2878.5, 1926.4)),
        ("baseline", 0.29, (1468.2, 1094.7, 2824.2, 1863.4)),
        ("+ Data Augmentation", 0.33, (1476.1, 1097.6, 2771.8, 2001.1)),
        ("+ Data Augmentation + Soft-NMS", 0.44, (1465.4, 1090.2, 2809.4, 1944.4)),
        ("+ Data Augmentation + Soft-NMS + WFF", 0.26, (1467.0, 974.8, 2830.1, 1923.2)),
        ("+ Data Augmentation + Soft-NMS + WFF + DLKA", 0.47, (1458.0, 1152.8, 2874.5, 1921.7)),
    ]
    export_box_style_icons(boxes, output_dir="result_output/icons", size_px=128, dpi=300)
