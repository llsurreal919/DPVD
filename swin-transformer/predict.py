import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# 导入模型
from model import swin_base_patch4_window7_224_in22k as create_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = 224
    data_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # 图像文件夹路径
    img_dir = "E:/swin-transformer/newn4_dataset_spilt/train/Vitiligo"
    assert os.path.exists(img_dir), f"Image directory '{img_dir}' does not exist."

    # 加载类别索引
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"File '{json_path}' does not exist."

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 自动初始化类别计数器
    class_count = {v: 0 for v in class_indict.values()}

    # 加载模型
    model = create_model(num_classes=2).to(device)
    model_weight_path = "E:/swin-transformer/train2/weights/model-49.pth"
    assert os.path.exists(model_weight_path), f"Weights file '{model_weight_path}' does not exist."
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # 遍历图像进行预测
    with torch.no_grad():
        for filename in os.listdir(img_dir):
            if filename.lower().endswith((".jpg", ".png")):
                img_path = os.path.join(img_dir, filename)
                img = Image.open(img_path).convert("RGB")
                img = data_transform(img)
                img = torch.unsqueeze(img, dim=0).to(device)

                output = torch.squeeze(model(img)).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).item()
                class_name = class_indict[str(predict_cla)]

                if class_name in class_count:
                    class_count[class_name] += 1
                else:
                    print(f"[Warning] Unexpected class name: {class_name}")

    # 打印分类结果
    print("📊 分类统计结果：")
    for cls_name, count in class_count.items():
        print(f"{cls_name}: {count}")


if __name__ == '__main__':
    main()
