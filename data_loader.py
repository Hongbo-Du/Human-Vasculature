from parameters import *
from torchvision import transforms
from pathlib import Path
import pandas as pd
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import torch.nn.functional as F


class DataLoader:
    def __init__(self):
        self.data_folder = Path(DATA_FOLDER)
        self.train_data_folder = Path(TRAIN_DATA_FOLDER)
        self.test_data_folder = Path(TEST_DATA_FOLDER)
        self.submission = Path(SUBMISSION_PATH)
        self.train_rles = Path(TRAIN_RLES_PATH)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # Add more transformations as needed
        ])

    def load_train_data(self, show=False, train_set=None):
        data = {}
        if train_set is not None:
            images_path = self.train_data_folder / train_set / 'images'
            labels_path = self.train_data_folder / train_set / 'labels'

            if Path(images_path).exists():
                images_path = images_path
            else:
                images_path = None

            if images_path is not None:
                image_files = list(images_path.iterdir())
                label_files = list(labels_path.iterdir())

                for img_file, label_file in zip(image_files, label_files):
                    img = self.load_image(str(img_file))
                    label = self.load_image(str(label_file))
                    if show is True:
                        self.show_image(str(img_file))
                    data[str(img_file.relative_to(self.data_folder))] = {'image': img, 'label': label}
            else:
                label_files = list(labels_path.iterdir())

                for label_file in label_files:
                    label = self.load_image(str(label_file))
                    if show is True:
                        self.show_image(str(label_file))
                    data[str(label_file.relative_to(self.data_folder))] = {'label': label}

        else:
            for kidney_folder in self.train_data_folder.iterdir():
                images_path = kidney_folder / 'images'
                labels_path = kidney_folder / 'labels'

                if Path(images_path).exists():
                    images_path = images_path
                else:
                    images_path = None

                if images_path is not None:
                    image_files = list(images_path.iterdir())
                    label_files = list(labels_path.iterdir())

                    for img_file, label_file in zip(image_files, label_files):
                        img = self.load_image(str(img_file))
                        label = self.load_image(str(label_file))
                        if show is True:
                            self.show_image(str(img_file))
                        data[str(img_file.relative_to(self.data_folder))] = {'image': img, 'label': label}
                else:
                    label_files = list(labels_path.iterdir())

                    for label_file in label_files:
                        label = self.load_image(str(label_file))
                        if show is True:
                            self.show_image(str(label_file))
                        data[str(label_file.relative_to(self.data_folder))] = {'label': label}

        return data

    def load_test_data(self, show=False):
        data = {}
        for kidney_folder in self.test_data_folder.iterdir():
            images_path = kidney_folder / 'images'
            image_files = list(images_path.iterdir())

            for img_file in image_files:
                img = self.load_image(str(img_file))
                if show is True:
                    self.show_image(str(img_file))
                data[str(img_file.relative_to(self.data_folder))] = {'image': img}

        return data

    def load_sample_submission(self):
        return pd.read_csv(self.submission)

    def load_train_rles(self):
        return pd.read_csv(self.train_rles)

    def load_image(self, file_path):
        img = tiff.imread(file_path).astype(np.float32)
        img = self.transform(img)
        img = F.normalize(img, p=2, dim=1)
        return img

    @staticmethod
    def show_image(file_path):
        img = tiff.imread(file_path)
        plt.imshow(img, cmap='gray')
        plt.title(f"{file_path[ file_path.find('data'):]}")
        plt.axis('off')
        plt.show()


if __name__ == "__main__":

    train = DataLoader().load_train_data(train_set=KIDNEY_3_DENSE)
    print(train)

    # test = DataLoader().load_test_data()
    # print(test['test\\kidney_6\\images\\0000.tif'])

    # sub = DataLoader().load_sample_submission()
    # print(sub)

    # rles = DataLoader().load_train_rles()
    # print(rles)


