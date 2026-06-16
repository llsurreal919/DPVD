import os

from PIL import Image
from PIL import ImageEnhance

img_dir = "E:/swin-transformer/newn4_dataset_spilt/train/Nonvitiligo"
for filename in os.listdir(img_dir):
    if filename.endswith(".jpg"):
        img_path = os.path.join(img_dir, filename)
        assert os.path.exists(img_path), "file: '{}' does not exist.".format(img_path)
        img = Image.open(img_path)
        enh_con = ImageEnhance.Contrast(img)
        contrast = 1.5
        img_contrasted = enh_con.enhance(contrast)
        output_dir = 'E:/swin-transformer/newn5_dataset_spilt/train/Nonvitiligo'
        img_contrasted.save(os.path.join(output_dir, filename))

