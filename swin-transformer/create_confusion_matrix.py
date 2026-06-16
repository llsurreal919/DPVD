import os
import json
import argparse
import sys

import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn.metrics import roc_curve, auc

from utils import read_split_data
from my_dataset import MyDataSet
from model import swin_base_patch4_window7_224_in22k as create_model


class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)

        with open("confusion_matrix.txt", "w") as f:
            f.write("Confusion Matrix:\n")
            for row in matrix.astype(int):
                f.write("\t".join(map(str, row)) + "\n")

        plt.imshow(matrix, cmap=plt.cm.Blues)
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        plt.yticks(range(self.num_classes), self.labels)
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion Matrix')

        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.savefig("confusion_matrix5.png", dpi=300)
        plt.close()


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    _, _, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = 384
    data_transform = {
        "val": transforms.Compose([
            transforms.Resize(int(img_size * 1.143)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
    }

    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes)
    assert os.path.exists(args.weights), f"cannot find {args.weights} file"
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)

    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), f"cannot find {json_label_path} file"
    with open(json_label_path, 'r') as f:
        class_indict = json.load(f)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=args.num_classes, labels=labels)

    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for step, val_data in enumerate(tqdm(val_loader, file=sys.stdout)):
            val_images, val_labels = val_data
            outputs = model(val_images.to(device))
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            confusion.update(preds.cpu().numpy(), val_labels.cpu().numpy())
            batch_labels = val_labels.cpu().numpy()
            batch_probs = probs.cpu().numpy()
            all_labels.extend(batch_labels)
            all_probs.extend(batch_probs)

            # 👇 打印前 3 步的数据
            if step < 3:
                print(f"\n[Step {step}]")
                print("val_labels:", batch_labels.tolist())
                print("probs:", batch_probs.tolist())

    confusion.plot()
    confusion.summary()

    # ✅ 限制计算 ROC 的最大样本数（加速调试）
    max_roc_points = 1000
    label_array = np.array(all_labels[:max_roc_points])
    prob_array = np.array(all_probs[:max_roc_points])

    fpr, tpr, thresholds = roc_curve(label_array, prob_array)
    roc_auc = auc(fpr, tpr)

    with open("roc_curve_data5.txt", "w") as f:
        f.write("FPR\tTPR\tThreshold\n")
        for i in range(len(fpr)):
            f.write(f"{fpr[i]:.4f}\t{tpr[i]:.4f}\t{thresholds[i]:.4f}\n")

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve of Swin Transformer')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("roc_curve5.png", dpi=300)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--data-path', type=str, default="E:/swin-transformer/newn4_dataset_spilt/test")
    parser.add_argument('--weights', type=str,
                        default="E:/swin-transformer/train5/weights/model-49.pth")
    parser.add_argument('--device', default='cuda:0', help='cuda:0 or cpu')

    opt = parser.parse_args()
    main(opt)