import torchvision.transforms.functional as F

class DynamicCenterCrop(object):
    def __call__(self, img):
        """
        Crop the largest possible square from the center of the image, while maintaining aspect ratio.
        """
        w, h = img.size
        crop_size = min(w, h)
        left = (w - crop_size) / 2
        top = (h - crop_size) / 2
        right = (w + crop_size) / 2
        bottom = (h + crop_size) / 2
        return F.crop(img, top, left, crop_size, crop_size)