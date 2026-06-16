import torch
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    # def generate(self, input_tensor, class_idx=None):
    #     self.model.eval()
    #     output = self.model(input_tensor)
    #
    #     if class_idx is None:
    #         class_idx = 1  # vitiligo 类别索引
    #
    #     # 分割模型输出维度：[B, C, H, W]，对目标类像素的响应求和作为 loss
    #     loss = output[:, class_idx, :, :].sum()
    #
    #     self.model.zero_grad()
    #     loss.backward(retain_graph=True)
    #
    #     weights = self.gradients.mean(dim=(2, 3), keepdim=True)
    #     cam = (weights * self.activations).sum(dim=1, keepdim=True)
    #     cam = torch.relu(cam)
    #     cam = cam.squeeze().cpu().numpy()
    #     cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
    #     cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
    #     return cam

    def generate(self, input_tensor, raw_image, class_idx=1):
        """
        input_tensor: shape [1, C, H, W] - preprocessed tensor
        raw_image: original RGB image (numpy array) in HxWx3 format
        class_idx: target class for CAM (default=1 for vitiligo)
        """
        self.model.eval()
        output = self.model(input_tensor)
        loss = output[:, class_idx, :, :].sum()

        self.model.zero_grad()
        loss.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()

        # Resize CAM to original image size
        raw_h, raw_w = raw_image.shape[:2]
        cam_resized = cv2.resize(cam, (raw_w, raw_h))

        # Normalize CAM
        cam_normalized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)

        # Threshold to determine if heatmap is meaningful
        if cam_normalized.max() < 0.2:  # e.g., activation too weak
            print("CAM activation too low, returning original image.")
            return raw_image

        # Apply colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_normalized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Overlay CAM on image
        overlay = np.float32(heatmap) * 0.5 + np.float32(raw_image) * 0.5
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        return overlay

    def clear_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
