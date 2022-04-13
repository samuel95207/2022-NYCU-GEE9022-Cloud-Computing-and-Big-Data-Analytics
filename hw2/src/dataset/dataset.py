from torch.utils.data.dataset import Dataset
from PIL import Image
from os import listdir
from os.path import isfile, isdir, join


class UnlabeledDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.img_filenames = [filename for filename in sorted(listdir(root)) if isfile(join(root, filename))]

    def __getitem__(self, index):
        imgpath = join(self.root, self.img_filenames[index])
        img = Image.open(imgpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, -1

    def __len__(self):
        return len(self.img_filenames)


class LabeledDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.img_filenames = []
        self.labels = []

        for folder in sorted(listdir(root)):
            folder_path = join(root, folder)
            if not isdir(folder_path):
                continue
            for filename in sorted(listdir(folder_path)):
                file_path = join(folder_path, filename)
                if not isfile(file_path):
                    continue
                self.img_filenames.append(file_path)
                self.labels.append(folder)



    def __getitem__(self, index):
        imgpath = self.img_filenames[index]
        img = Image.open(imgpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = int(self.labels[index])
        return img, label

    def __len__(self):
        return len(self.img_filenames)
