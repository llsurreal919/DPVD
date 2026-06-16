import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def calculate_iou(boxA, boxB):
    """计算 IoU 并返回交集坐标"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = float(boxAArea + boxBArea - interArea)
    iou = interArea / unionArea if unionArea > 0 else 0
    return iou, (xA, yA, xB, yB)


def save_legend_only_image(handles, output_path, items_per_row=3, force_rows=2):
    """
    保存只包含图例的透明 PNG。
    - items_per_row: 每行显示的图例个数
    - force_rows:    固定行数（不足会补空格子）
    """
    import matplotlib.patches as mpatches

    target_total = items_per_row * force_rows
    # 补齐透明占位，保证布局完整
    while len(handles) < target_total:
        handles.append(
            mpatches.Patch(edgecolor='none', facecolor='none', label="")
        )

    fig_width = max(6, items_per_row * 2.8)
    fig_height = force_rows * 1.4

    fig_legend = plt.figure(figsize=(fig_width, fig_height))
    fig_legend.legend(
        handles=handles,
        loc='center', ncol=items_per_row,
        frameon=False, fontsize=11,
        handlelength=2.5, handletextpad=1.2,
        borderpad=0.8
    )
    plt.axis('off')
    plt.tight_layout()
    fig_legend.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig_legend)
    print(f"✅ Legend 图像已保存至: {output_path}")


def draw_boxes(image_path, box_list, output_path=None):
    if len(box_list) < 2:
        raise ValueError("❌ 至少需要一个 GT 框和一个预测框")

    gt_label, gt_conf, gt_box = box_list[0]
    pred_boxes = box_list[1:]
    final_label, final_conf, final_box = box_list[-1]

    colormap = plt.get_cmap("tab10")
    legend_handles = []

    img = Image.open(image_path).convert("RGB")
    img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))

    # 绘制 GT 框
    ax.add_patch(patches.Rectangle(
        (gt_box[0], gt_box[1]), gt_box[2] - gt_box[0], gt_box[3] - gt_box[1],
        linewidth=3.5, edgecolor='black', linestyle='--', facecolor='none'
    ))
    legend_handles.append(patches.Patch(
        edgecolor='black', facecolor='none', linestyle='--', linewidth=2, label="GT"
    ))

    # 绘制预测框
    for i, (label, conf, coords) in enumerate(pred_boxes):
        _ = calculate_iou(gt_box, coords)
        rect = patches.Rectangle(
            (coords[0], coords[1]), coords[2] - coords[0], coords[3] - coords[1],
            linewidth=2, edgecolor=colormap(i), facecolor='none'
        )
        ax.add_patch(rect)
        legend_handles.append(patches.Patch(
            edgecolor=colormap(i), facecolor='none', linewidth=2, label=label
        ))

    # 绘制重叠区域
    _, (ix1, iy1, ix2, iy2) = calculate_iou(gt_box, final_box)
    ax.add_patch(patches.Rectangle(
        (ix1, iy1), ix2 - ix1, iy2 - iy1,
        facecolor=colormap(len(pred_boxes) - 1), edgecolor='none', alpha=0.3
    ))

    ax.axis("off")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✅ 图像已保存到: {output_path}")

        # 保存 Legend：固定两行三列
        legend_path = output_path.replace(".png", "_legend.png")
        save_legend_only_image(legend_handles, legend_path, items_per_row=3, force_rows=2)
    else:
        plt.show()


if __name__ == "__main__":
    boxes = [
        ("gt", None, (1471.2, 1165.6, 2878.5, 1926.4)),
        ("baseline", 0.29, (1468.2, 1094.7, 2824.2, 1863.4)),
        ("+ Data Augmentation", 0.33, (1476.1, 1097.6, 2771.8, 2001.1)),
        ("+ Data Augmentation + Soft-NMS", 0.44, (1465.4, 1090.2, 2809.4, 1944.4)),
        ("+ Data Augmentation + Soft-NMS + WFF", 0.26, (1467.0, 974.8, 2830.1, 1923.2)),
        ("+ Data Augmentation + Soft-NMS + WFF + DLKA", 0.47, (1458.0, 1152.8, 2874.5, 1921.7)),
    ]

    draw_boxes(
        image_path="../test_data/img_t_10_0025.jpg",
        box_list=boxes,
        output_path="result_output/in_code_box_compare6.png"
    )
