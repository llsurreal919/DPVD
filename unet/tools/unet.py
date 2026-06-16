import colorsys
import copy
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
# from nets.unetpp import Unet as unet
from nets.unet import Unet as unet
from utils.utils import cvtColor, preprocess_input, resize_image, show_config
from utils.gradcam import GradCAM  # 假设你放在 utils 文件夹中

from sklearn.metrics import roc_auc_score
from PIL import Image as PILImage

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
        return 0.5  # 如果标签全为1或0，无法计算AUC


# --------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和num_classes都需要修改！
#   如果出现shape不匹配
#   一定要注意训练时的model_path和num_classes数的修改
# --------------------------------------------#
class Unet(object):
    _defaults = {
        # -------------------------------------------------------------------#
        #   model_path指向logs文件夹下的权值文件
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表miou较高，仅代表该权值在验证集上泛化性能较好。
        # -------------------------------------------------------------------#
        "model_path": 'logs/Unet(pretrain_vgg_Adam_expansion)/best_epoch_weights.pth',
        # --------------------------------#
        #   所需要区分的类的个数+1
        # --------------------------------#
        "num_classes": 2,
        # --------------------------------#
        #   所使用的的主干网络：vgg、resnet50   
        # --------------------------------#
        "backbone": "vgg",
        # --------------------------------#
        #   输入图片的大小
        # --------------------------------#
        "input_shape": [512, 512],
        # -------------------------------------------------#
        #   mix_type参数用于控制检测结果的可视化方式
        #
        #   mix_type = 0的时候代表原图与生成的图进行混合
        #   mix_type = 1的时候代表仅保留生成的图
        #   mix_type = 2的时候代表仅扣去背景，仅保留原图中的目标
        # -------------------------------------------------#
        "mix_type": 0,
        # --------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # --------------------------------#
        "cuda": True,
    }

    # ---------------------------------------------------#
    #   初始化UNET
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        if self.num_classes <= 21:
            self.colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                           (0, 128, 128),
                           (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128),
                           (192, 0, 128),
                           (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                           (0, 64, 128),
                           (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        # ---------------------------------------------------#
        #   获得模型
        # ---------------------------------------------------#
        self.generate()

        show_config(**self._defaults)

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def generate(self, onnx=False):
        self.net = unet(num_classes=self.num_classes, backbone=self.backbone)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image, count=False, name_classes=None):
        from PIL import Image as PILImage
        import os

        # === ✅ 获取原始路径（避免被 fromarray 覆盖） ===
        filename = getattr(image, "filename", None)

        # === 图像预处理 ===
        image = cvtColor(image)
        old_img = copy.deepcopy(image)
        orininal_h, orininal_w = np.array(image).shape[:2]

        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            pred_mask = pr.argmax(axis=-1)

        # === 可视化 ===
        if self.mix_type == 0:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pred_mask, [-1])],
                                 [orininal_h, orininal_w, -1])
            image = PILImage.fromarray(np.uint8(seg_img))
            image = PILImage.blend(old_img, image, 0.7)
        elif self.mix_type == 1:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pred_mask, [-1])],
                                 [orininal_h, orininal_w, -1])
            image = PILImage.fromarray(np.uint8(seg_img))
        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pred_mask != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
            image = PILImage.fromarray(np.uint8(seg_img))

        # === 计数 ===
        if count:
            total_points_num = orininal_h * orininal_w
            print('-' * 63)
            print("|%25s | %15s | %15s|" % ("Key", "Value", "Ratio"))
            print('-' * 63)
            for i in range(self.num_classes):
                num = np.sum(pred_mask == i)
                ratio = num / total_points_num * 100
                if num > 0:
                    print("|%25s | %15s | %14.2f%%|" % (str(name_classes[i]), str(num), ratio))
                    print('-' * 63)

        # === ✅ 初始化指标 ===
        d = j = sen = pre = auc_val = None

        # === ✅ 计算指标 ===
        try:
            if filename:
                basename = os.path.splitext(os.path.basename(filename))[0]
                gt_path = os.path.join("datasets/SegmentationClass", basename + ".png")
                print(f"[DEBUG] Looking for GT mask: {gt_path}")
                if os.path.exists(gt_path):
                    gt = np.array(PILImage.open(gt_path).resize((orininal_w, orininal_h)))
                    # 把掩码 resize 成模型预测图一样大
                    gt = np.array(PILImage.open(gt_path).resize((orininal_w, orininal_h), Image.NEAREST))


                    print(f"[DEBUG] Loaded GT shape: {gt.shape}")
                    if np.max(gt) > 1:
                        gt = (gt > 127).astype(np.uint8)

                    y_true = (gt == 1).astype(np.uint8)
                    y_pred = (pred_mask == 1).astype(np.uint8)
                    y_prob = pr[..., 1] if pr.ndim == 3 and pr.shape[-1] > 1 else y_pred

                    # ✅ 可视化 GT 掩码（绿色填充 + 轮廓）
                    gt_mask = (y_true * 255).astype(np.uint8)

                    # 创建颜色图层
                    gt_color = np.zeros((orininal_h, orininal_w, 3), dtype=np.uint8)
                    gt_color[..., 1] = gt_mask  # G 通道 = 绿色

                    # 半透明融合
                    image_np = np.array(image.convert("RGB"))
                    overlay = cv2.addWeighted(image_np, 1.0, gt_color, 0.4, 0)

                    # 加上轮廓线
                    contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay, contours, -1, (0, 255, 0), thickness=2)

                    image = PILImage.fromarray(overlay)

                    # ✅ 可视化预测掩码（红色填充 + 轮廓）
                    pred_mask_bin = (y_pred * 255).astype(np.uint8)

                    # 创建红色图层（仅 R 通道）
                    pred_color = np.zeros((orininal_h, orininal_w, 3), dtype=np.uint8)
                    pred_color[..., 0] = pred_mask_bin  # R 通道 = 红色

                    # 半透明融合在已有叠加图上（GT已经是绿色）
                    overlay = cv2.addWeighted(np.array(image), 1.0, pred_color, 0.4, 0)

                    # 加上红色边界线
                    contours_pred, _ = cv2.findContours(pred_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay, contours_pred, -1, (255, 0, 0), thickness=2)

                    # 转为 PIL 图像
                    image = PILImage.fromarray(overlay)

                    d = dice_coefficient(y_true, y_pred)
                    j = iou_score(y_true, y_pred)
                    sen = sensitivity(y_true, y_pred)
                    pre = precision(y_true, y_pred)
                    auc_val = auc_score(y_true, y_prob)

                    print(f"[{basename}]")
                    print(f"  Dice:        {d:.4f}")
                    print(f"  IoU (Jacc.): {j:.4f}")
                    print(f"  Sensitivity: {sen:.4f}")
                    print(f"  Precision:   {pre:.4f}")
                    print(f"  AUC:         {auc_val:.4f}")
                else:
                    print(f"[WARNING] GT not found: {gt_path}")
            else:
                print("[DEBUG] Skip metric computation: image has no filename")
        except Exception as e:
            print(f"[WARNING] Failed to compute metrics: {e}")

        return image, d, j, sen, pre, auc_val


    # def detect_image(self, image, count=False, name_classes=None):
    #     from PIL import Image as PILImage
    #     import os
    #
    #     # === 图像预处理 === #
    #     image = cvtColor(image)
    #     old_img = copy.deepcopy(image)
    #     orininal_h, orininal_w = np.array(image).shape[:2]
    #
    #     image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
    #     image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
    #
    #     with torch.no_grad():
    #         images = torch.from_numpy(image_data)
    #         if self.cuda:
    #             images = images.cuda()
    #         pr = self.net(images)[0]
    #         pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
    #         pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
    #              int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
    #         pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
    #         pred_mask = pr.argmax(axis=-1)
    #
    #     # === 可视化 === #
    #     if self.mix_type == 0:
    #         seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pred_mask, [-1])],
    #                              [orininal_h, orininal_w, -1])
    #         image = PILImage.fromarray(np.uint8(seg_img))
    #         image = PILImage.blend(old_img, image, 0.7)
    #     elif self.mix_type == 1:
    #         seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pred_mask, [-1])],
    #                              [orininal_h, orininal_w, -1])
    #         image = PILImage.fromarray(np.uint8(seg_img))
    #     elif self.mix_type == 2:
    #         seg_img = (np.expand_dims(pred_mask != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
    #         image = PILImage.fromarray(np.uint8(seg_img))
    #
    #     # === 计数打印 === #
    #     if count:
    #         total_points_num = orininal_h * orininal_w
    #         print('-' * 63)
    #         print("|%25s | %15s | %15s|" % ("Key", "Value", "Ratio"))
    #         print('-' * 63)
    #         for i in range(self.num_classes):
    #             num = np.sum(pred_mask == i)
    #             ratio = num / total_points_num * 100
    #             if num > 0:
    #                 print("|%25s | %15s | %14.2f%%|" % (str(name_classes[i]), str(num), ratio))
    #                 print('-' * 63)
    #
    #     # ========== 初始化 ==========
    #     d, j, sen, pre, auc_val = None, None, None, None, None
    #
    #     try:
    #         image_filename = getattr(image, "filename", None)
    #         if image_filename:
    #             basename = os.path.splitext(os.path.basename(image_filename))[0]
    #             gt_path = os.path.join("datasets/SegmentationClass", basename + ".png")
    #             print(f"[DEBUG] Looking for GT mask: {gt_path}")
    #             if os.path.exists(gt_path):
    #                 # 加载GT并resize
    #                 gt = np.array(PILImage.open(gt_path).resize((orininal_w, orininal_h)))
    #                 print("gt: ")
    #                 print(gt)
    #                 if np.max(gt) > 1:
    #                     gt = (gt > 127).astype(np.uint8)  # 转为0/1标签
    #
    #                 y_true = (gt == 1).astype(np.uint8)
    #                 y_pred = (pr == 1).astype(np.uint8)
    #                 y_prob = pr[..., 1] if pr.ndim == 3 and pr.shape[-1] > 1 else y_pred
    #
    #                 # 计算指标
    #                 d = dice_coefficient(y_true, y_pred)
    #                 j = iou_score(y_true, y_pred)
    #                 sen = sensitivity(y_true, y_pred)
    #                 pre = precision(y_true, y_pred)
    #                 auc_val = auc_score(y_true, y_prob)
    #
    #                 print(f"[{basename}]")
    #                 print(f"  Dice:        {d:.4f}")
    #                 print(f"  IoU (Jacc.): {j:.4f}")
    #                 print(f"  Sensitivity: {sen:.4f}")
    #                 print(f"  Precision:   {pre:.4f}")
    #                 print(f"  AUC:         {auc_val:.4f}")
    #             else:
    #                 print(f"[WARNING] GT not found: {gt_path}")
    #     except Exception as e:
    #         print(f"[WARNING] Failed to compute metrics: {e}")
    #
    #     return image, d, j, sen, pre, auc_val

    def get_FPS(self, image, test_interval):
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            # ---------------------------------------------------#
            #   图片传入网络进行预测
            # ---------------------------------------------------#
            pr = self.net(images)[0]
            # ---------------------------------------------------#
            #   取出每一个像素点的种类
            # ---------------------------------------------------#
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
            # --------------------------------------#
            #   将灰条部分截取掉
            # --------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                # ---------------------------------------------------#
                #   图片传入网络进行预测
                # ---------------------------------------------------#
                pr = self.net(images)[0]
                # ---------------------------------------------------#
                #   取出每一个像素点的种类
                # ---------------------------------------------------#
                pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
                # --------------------------------------#
                #   将灰条部分截取掉
                # --------------------------------------#
                pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                     int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        im = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        input_layer_names = ["images"]
        output_layer_names = ["output"]

        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                          im,
                          f=model_path,
                          verbose=False,
                          opset_version=12,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True,
                          input_names=input_layer_names,
                          output_names=output_layer_names,
                          dynamic_axes=None)

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))

    def get_miou_png(self, image):
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            # ---------------------------------------------------#
            #   图片传入网络进行预测
            # ---------------------------------------------------#
            pr = self.net(images)[0]
            # ---------------------------------------------------#
            #   取出每一个像素点的种类
            # ---------------------------------------------------#
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            # --------------------------------------#
            #   将灰条部分截取掉
            # --------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            # ---------------------------------------------------#
            #   进行图片的resize
            # ---------------------------------------------------#
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            # ---------------------------------------------------#
            #   取出每一个像素点的种类
            # ---------------------------------------------------#
            pr = pr.argmax(axis=-1)

        image = Image.fromarray(np.uint8(pr))
        return image

    def visualize_cam(self, image_path, save_path='outputcam/cam.jpg'):
        from utils.gradcam import GradCAM
        from PIL import Image
        import numpy as np
        import torch
        import cv2

        # 读取并保留原图
        image = Image.open(image_path).convert('RGB')
        raw_image = np.array(image)

        # resize + 归一化 + 转 tensor
        image_data = np.array(image.resize((self.input_shape[1], self.input_shape[0])))
        image_data = np.expand_dims(np.transpose(image_data / 255.0, (2, 0, 1)), 0).astype(np.float32)
        input_tensor = torch.from_numpy(image_data).cuda() if self.cuda else torch.from_numpy(image_data)

        # 适配 DataParallel
        net_module = self.net.module if isinstance(self.net, torch.nn.DataParallel) else self.net

        # 选择 Grad-CAM 的目标层：无论是 VGG 还是 ResNet 都统一 hook 到 decoder
        if hasattr(net_module, 'up_concat1') and hasattr(net_module.up_concat1, 'conv2'):
            target_layer = net_module.up_concat1.conv2
        else:
            raise ValueError("Cannot find up_concat1.conv2 in the current model.")

        # 构建 Grad-CAM
        cam = GradCAM(self.net, target_layer)
        result = cam.generate(input_tensor, raw_image)
        cam.clear_hooks()

        # 保存可视化图
        cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"Grad-CAM saved to {save_path}")
