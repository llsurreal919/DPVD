import os

import matplotlib
import torch
import torch.nn.functional as F

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal

import cv2
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .utils import cvtColor, preprocess_input, resize_image
from .utils_metrics import compute_mIoU


class LossHistory():
    def __init__(self, log_dir, model, input_shape, val_loss_flag=True):
        self.log_dir        = log_dir
        self.val_loss_flag  = val_loss_flag

        self.losses         = []
        if self.val_loss_flag:
            self.val_loss   = []

        os.makedirs(self.log_dir)
        self.writer     = SummaryWriter(self.log_dir)
        try:
            dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss = None):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        if self.val_loss_flag:
            self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        if self.val_loss_flag:
            with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
                f.write(str(val_loss))
                f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        if self.val_loss_flag:
            self.writer.add_scalar('val_loss', val_loss, epoch)

        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        if self.val_loss_flag:
            plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')

        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            if self.val_loss_flag:
                plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")


class EvalCallback():
    def __init__(self, net, input_shape, num_classes, image_ids, dataset_path, log_dir, cuda, \
                 miou_out_path=".temp_miou_out", eval_flag=True, period=1):
        super(EvalCallback, self).__init__()

        self.net = net
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.image_ids = [image_id.split()[0] for image_id in image_ids]
        self.dataset_path = dataset_path
        self.log_dir = log_dir
        self.cuda = cuda
        self.miou_out_path = miou_out_path
        self.eval_flag = eval_flag
        self.period = period

        # 保存每个epoch的mIoU和Dice，便于后续作图
        self.mious = [0]
        self.dices = [0]  # 新增，用于记录Dice值
        self.epoches = [0]

        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(0) + "\n")
            with open(os.path.join(self.log_dir, "epoch_dice.txt"), 'a') as f:  # 新增保存Dice的文件
                f.write(str(0) + "\n")

    def get_miou_png(self, image):
        # 将图像转换成RGB
        image = cvtColor(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        # 通过添加灰条实现不失真resize
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        # 添加batch维度并预处理
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # 图片传入网络进行预测
            pr = self.net(images)[0]
            # 进行softmax，并移动至CPU
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            # 截取灰条部分
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            # Resize回原始尺寸
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            # 取每个像素点的类别
            pr = pr.argmax(axis=-1)

        image = Image.fromarray(np.uint8(pr))
        return image

    from sklearn.metrics import roc_auc_score

    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            gt_dir = os.path.join(self.dataset_path, "VOC2007/SegmentationClass/")
            pred_dir = os.path.join(self.miou_out_path, 'detection-results')
            if not os.path.exists(self.miou_out_path):
                os.makedirs(self.miou_out_path)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)

            print("Get predictions for evaluation metrics.")
            all_dices, all_ious = [], []
            all_precisions, all_sensitivities, all_aucs = [], [], []

            for image_id in tqdm(self.image_ids):
                image_path = os.path.join(self.dataset_path, "VOC2007/JPEGImages/" + image_id + ".jpg")
                label_path = os.path.join(self.dataset_path, "VOC2007/SegmentationClass/" + image_id + ".png")

                # Load original image and ground truth
                image = Image.open(image_path)
                label = np.array(Image.open(label_path).resize(image.size))  # GT
                y_true = (label == 1).astype(np.uint8)

                # Predict
                image_data = self.get_miou_png(image)
                image_data.save(os.path.join(pred_dir, image_id + ".png"))
                y_pred = np.array(image_data)
                y_pred_bin = (y_pred == 1).astype(np.uint8)

                # Softmax second channel as probability map (for AUC)
                with torch.no_grad():
                    input_image, nw, nh = resize_image(cvtColor(image), self.input_shape)
                    input_tensor = np.expand_dims(
                        np.transpose(preprocess_input(np.array(input_image, np.float32)), (2, 0, 1)), 0)
                    input_tensor = torch.from_numpy(input_tensor).cuda() if self.cuda else torch.from_numpy(
                        input_tensor)
                    output = self.net(input_tensor)[0]
                    prob_map = F.softmax(output.permute(1, 2, 0), dim=-1).cpu().numpy()
                    prob_map = prob_map[int((self.input_shape[0] - nh) // 2):int((self.input_shape[0] - nh) // 2 + nh),
                               int((self.input_shape[1] - nw) // 2):int((self.input_shape[1] - nw) // 2 + nw)]
                    prob_map = cv2.resize(prob_map, image.size, interpolation=cv2.INTER_LINEAR)
                    y_prob = prob_map[..., 1]

                def dice_coefficient(y_true, y_pred):
                    smooth = 1e-6
                    intersection = np.sum(y_true * y_pred)
                    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

                def iou_score(y_true, y_pred):
                    smooth = 1e-6
                    intersection = np.sum(y_true * y_pred)
                    union = np.sum(y_true) + np.sum(y_pred) - intersection
                    return (intersection + smooth) / (union + smooth)

                def sensitivity(y_true, y_pred):
                    tp = np.sum((y_true == 1) & (y_pred == 1))
                    fn = np.sum((y_true == 1) & (y_pred == 0))
                    return tp / (tp + fn + 1e-6)

                def precision(y_true, y_pred):
                    tp = np.sum((y_true == 1) & (y_pred == 1))
                    fp = np.sum((y_true == 0) & (y_pred == 1))
                    return tp / (tp + fp + 1e-6)

                def auc_score(y_true, y_prob):
                    try:
                        return roc_auc_score(y_true.flatten(), y_prob.flatten())
                    except:
                        return 0.5  # Return neutral value if only one class

                # Compute metrics
                all_dices.append(dice_coefficient(y_true, y_pred_bin))
                all_ious.append(iou_score(y_true, y_pred_bin))
                all_precisions.append(precision(y_true, y_pred_bin))
                all_sensitivities.append(sensitivity(y_true, y_pred_bin))
                all_aucs.append(auc_score(y_true, y_prob))

            # Average metrics
            temp_miou = np.nanmean(all_ious) * 100
            temp_dice = np.nanmean(all_dices) * 100
            temp_precision = np.nanmean(all_precisions) * 100
            temp_sensitivity = np.nanmean(all_sensitivities) * 100
            temp_auc = np.nanmean(all_aucs) * 100

            # Save all
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(f"{temp_miou:.4f}\n")
            with open(os.path.join(self.log_dir, "epoch_dice.txt"), 'a') as f:
                f.write(f"{temp_dice:.4f}\n")
            with open(os.path.join(self.log_dir, "epoch_precision.txt"), 'a') as f:
                f.write(f"{temp_precision:.4f}\n")
            with open(os.path.join(self.log_dir, "epoch_sensitivity.txt"), 'a') as f:
                f.write(f"{temp_sensitivity:.4f}\n")
            with open(os.path.join(self.log_dir, "epoch_auc.txt"), 'a') as f:
                f.write(f"{temp_auc:.4f}\n")

            # Plot summary
            self.epoches.append(epoch)
            self.mious.append(temp_miou)
            self.dices.append(temp_dice)

            plt.figure()
            plt.plot(self.epoches, self.mious, 'red', linewidth=2, label='mIoU')
            plt.plot(self.epoches, self.dices, 'blue', linewidth=2, label='Dice')
            plt.plot(self.epoches, [temp_precision] * len(self.epoches), 'green', linestyle='--', label='Precision')
            plt.plot(self.epoches, [temp_sensitivity] * len(self.epoches), 'purple', linestyle='--',
                     label='Sensitivity')
            plt.plot(self.epoches, [temp_auc] * len(self.epoches), 'orange', linestyle='--', label='AUC')
            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Metric (%)')
            plt.title('Segmentation Metrics')
            plt.legend(loc="upper right")
            plt.savefig(os.path.join(self.log_dir, "epoch_metrics.png"))
            plt.close()

            shutil.rmtree(self.miou_out_path)
            print("All metrics calculated and saved.")

        def dice_coefficient(y_true, y_pred):
            smooth = 1e-6
            intersection = np.sum(y_true * y_pred)
            return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

        def iou_score(y_true, y_pred):
            smooth = 1e-6
            intersection = np.sum(y_true * y_pred)
            union = np.sum(y_true) + np.sum(y_pred) - intersection
            return (intersection + smooth) / (union + smooth)

        def sensitivity(y_true, y_pred):
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            return tp / (tp + fn + 1e-6)

        def precision(y_true, y_pred):
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            return tp / (tp + fp + 1e-6)

        from sklearn.metrics import roc_auc_score

        def auc_score(y_true, y_prob):
            y_true_flat = y_true.flatten()
            y_prob_flat = y_prob.flatten()
            return roc_auc_score(y_true_flat, y_prob_flat)
