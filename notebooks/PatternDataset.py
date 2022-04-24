from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import numpy as np


class PatternDataset(Dataset):
    def __init__(self, root_dir, transform=None, build_classification=False, name_cla='output.cla'):
        self.root_dir = root_dir
        self.transform = transform
        self.namefiles = []

        self.classes = sorted(os.listdir(self.root_dir))

        for cl in self.classes:
            for pat in os.listdir(os.path.join(self.root_dir, cl)):
                self.namefiles.append((pat, cl))

        print(f'TrainVal:{len(self.namefiles)}')
        self.namefiles = sorted(self.namefiles, key=lambda x: x[0])

        if build_classification:
            dictClasses = dict()

            for cl in self.classes:
                dictClasses[cl] = []

            for index, (name, cl) in enumerate(self.namefiles):
                dictClasses[cl].append((name, index))

            with open(name_cla, 'w') as f:
                f.write('PSB 1\n')
                f.write(f'{len(self.classes)} {len(self.namefiles)}\n')
                f.write('\n')
                for cl in self.classes:
                    f.write(f'{cl} 0 {len(dictClasses[cl])}\n')
                    for item in dictClasses[cl]:
                        f.write(f'{item[1]}\n')
                    f.write('\n')

    def __len__(self):
        return len(self.namefiles)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img_name = os.path.join(self.root_dir, self.namefiles[index][1], self.namefiles[index][0])
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return self.namefiles[index], image


if __name__ == "__main__":
    path_dataset = '/home/ivan/Documentos/DCC/Research/KunischPatterns/KunischDataset/test'

    dataset = PatternDataset(root_dir=path_dataset, build_classification=True, name_cla='patternTest.cla')

    print(dataset[0])