import os
import pathlib
import numpy as np
import cv2
import torch
import torch.utils.data as Data
from torchvision import transforms
from natsort import natsorted

def get_file_list(folder_path: str, p_postfix=['.jpg'], sub_dir=True) -> list:
    """
    Lấy danh sách file trong folder_path với các đuôi p_postfix.
    """
    assert os.path.exists(folder_path) and os.path.isdir(folder_path), f"{folder_path} not found!"
    if isinstance(p_postfix, str):
        p_postfix = [p_postfix]

    file_list = []
    if sub_dir:
        for rootdir, _, files in os.walk(folder_path):
            for file in files:
                for p in p_postfix:
                    if file.endswith(p):
                        file_list.append(os.path.join(rootdir, file))
    else:
        for file in os.listdir(folder_path):
            for p in p_postfix:
                if file.endswith(p):
                    file_list.append(os.path.join(folder_path, file))
    return natsorted(file_list)


class ImageData(Data.Dataset):
    """
    Dataset cho DocUnet: 
    - images: normalize về [-1,1] (có thể thay đổi)
    - labels: normalize về [0,1]
    """

    def __init__(self, img_root, image_size=(512,512), channel=3, transform=None, t_transform=None):
        """
        :param img_root: thư mục gốc chứa 'images' và 'labels'
        :param image_size: (H,W)
        :param channel: số channel của ảnh
        :param transform: torchvision transform cho image
        :param t_transform: transform cho label
        """
        self.channel = channel
        self.image_size = image_size

        # Lấy danh sách ảnh
        images_folder = os.path.join(img_root, 'images')
        self.image_path = get_file_list(images_folder, p_postfix=['.jpg'], sub_dir=True)
        self.image_path = [x for x in self.image_path if pathlib.Path(x).stat().st_size > 0]

        # Lấy danh sách label
        label_folder = os.path.join(img_root, 'labels')
        self.label_path = [os.path.join(label_folder, os.path.splitext(os.path.basename(x))[0]+'.npy') 
                           for x in self.image_path]

        # Transform
        # Nếu không truyền transform, dùng mặc định: image [-1,1], label [0,1]
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),  # 0-255 -> 0-1
                transforms.Normalize(mean=[0.5]*self.channel, std=[0.5]*self.channel)  # 0-1 -> -1->1
            ])
        else:
            self.transform = transform

        if t_transform is None:
            self.t_transform = transforms.ToTensor()  # 0-255 -> 0-1
        else:
            self.t_transform = t_transform

    def __getitem__(self, index):
        # Load image
        image = cv2.imread(self.image_path[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if self.channel==3 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, self.image_size)
        image = self.transform(image)

        # Load label
        label = np.load(self.label_path[index])
        label = cv2.resize(label, self.image_size, interpolation=cv2.INTER_NEAREST)
        label = self.t_transform(label)

        return image, label

    def __len__(self):
        return len(self.image_path)
