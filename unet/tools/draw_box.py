import cv2
import os

# 每个模型用不同颜色（BGR）
colors = [
    (255, 0, 0),    # 蓝色 - 模型1
    (0, 255, 0),    # 绿色 - 模型2
    (0, 0, 255),    # 红色 - 模型3
    (0, 255, 255),  # 黄色 - 模型4
    (255, 0, 255),  # 粉色 - 模型5
]

# 每个模型结果文件路径（每个文件中一行一个框）
model_result_files = [
    "results/model1.txt",
    "results/model2.txt",
    "results/model3.txt",
    "results/model4.txt",
    "results/model5.txt",
]

# 图像路径
image_path = "datasets/JPEGImages/img_t_10_0002.jpg"
output_path = "result_output/image1_compare.jpg"

# 加载图像
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Cannot load image {image_path}")

# 逐个模型读取并绘制框
for i, file_path in enumerate(model_result_files):
    color = colors[i]
    if not os.path.exists(file_path):
        print(f"Warning: File not found {file_path}")
        continue

    with open(file_path, "r") as f:
        for line in f:
            try:
                # 解析格式：class: 0, conf: 0.34, box: (633.4, 676.7, 2769.1, 2940.8)
                parts = line.strip().split(",")
                cls = parts[0].split(":")[1].strip()
                conf = float(parts[1].split(":")[1])
                box_str = parts[2].split(":")[1].strip().strip("()")
                x1, y1, x2, y2 = map(float, box_str.split())

                # 画框
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2)

                # 添加标签（模型编号 + 置信度）
                label = f"M{i+1}: {conf:.2f}"
                cv2.putText(image, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            except Exception as e:
                print(f"Failed to parse line: {line.strip()}\nError: {e}")

# 保存绘图结果
cv2.imwrite(output_path, image)
print(f"Saved comparison image to {output_path}")
