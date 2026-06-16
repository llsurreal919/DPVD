import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def calculate_iou(boxA, boxB):
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


def save_legend_only_image(handles, output_path, columns=1):
    fig_legend = plt.figure(figsize=(6, len(handles) * 0.35))  # 控制图像比例：细长
    fig_legend.legend(
        handles=handles,
        loc='center', ncol=columns,
        frameon=True, fontsize=10,
        framealpha=0.85, fancybox=True, borderpad=1.0
    )
    fig_legend.tight_layout(pad=1)
    fig_legend.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig_legend)
    print(f"✅ 图注图像已单独保存至: {output_path}")


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

    # Draw GT box
    ax.add_patch(patches.Rectangle(
        (gt_box[0], gt_box[1]), gt_box[2] - gt_box[0], gt_box[3] - gt_box[1],
        linewidth=3.5, edgecolor='black', linestyle='--', facecolor='none'
    ))
    conf_text = f"conf={gt_conf:.2f}" if gt_conf is not None else "conf=N/A"
    legend_handles.append(patches.Patch(
        edgecolor='black', facecolor='none', linestyle='--', linewidth=3.5,
        label=f"{gt_label} ({conf_text}) | GT"
    ))

    # Draw prediction boxes
    for i, (label, conf, coords) in enumerate(pred_boxes):
        iou, _ = calculate_iou(gt_box, coords)
        conf_text = f"conf={conf:.2f}" if conf is not None else "conf=N/A"
        display_label = f"{label} ({conf_text}) | IoU={iou:.3f}"

        rect = patches.Rectangle(
            (coords[0], coords[1]), coords[2] - coords[0], coords[3] - coords[1],
            linewidth=2, edgecolor=colormap(i), facecolor='none'
        )
        ax.add_patch(rect)
        legend_handles.append(patches.Patch(
            edgecolor=colormap(i), facecolor='none', linewidth=2, label=display_label
        ))

    # Draw overlap area
    _, (ix1, iy1, ix2, iy2) = calculate_iou(gt_box, final_box)
    ax.add_patch(patches.Rectangle(
        (ix1, iy1), ix2 - ix1, iy2 - iy1,
        facecolor=colormap(len(pred_boxes) - 1), edgecolor='none', alpha=0.3
    ))

    # ❌ 不显示 legend 图标在图上
    ax.axis("off")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✅ 图像已保存到: {output_path}")

        # ✅ 保存 legend 为单独细长图片
        legend_path = output_path.replace(".png", "_legend.png")
        save_legend_only_image(legend_handles, legend_path)
    else:
        plt.show()


if __name__ == "__main__":
    boxes = [
        ("gt", None, (704.5, 667.9, 2743.2, 2920.5)),
        ("baseline", 0.34, (633.4, 676.7, 2769.1, 2940.8)),
        ("+ A", 0.31, (622.2, 548.0, 2773.9, 3006.5)),
        ("+ A + B", 0.72, (712.7, 697.2, 2820.6, 2950.0)),
        ("+ A + B + C", 0.52, (707.7, 716.3, 2811.4, 2951.6)),
        ("+ A + B + C + D", 0.54, (692.6, 687.0, 2712.3, 2946.2)),
    ]

    draw_boxes(
        image_path="../test_data/img_t_10_0002.jpg",
        box_list=boxes,
        output_path="result_output/in_code_box_compare6.png"
    )
