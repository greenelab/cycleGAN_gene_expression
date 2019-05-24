import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class GeneExpressionDataset(Dataset):
    def __init__(self, root, mode="train"):
        # Training data loader
        data_file = os.path.join("../../data/%s" % opt.dataset_name +
                                 "/train/A/all-pseudomonas-gene-normalized.zip")

        rnaseq = pd.read_table(data_file, index_col=0, header=0).T

        #self.transform = transforms.Compose(transforms_)
        #self.unaligned = unaligned

        # self.files_A = sorted(
        #    glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*"))
        # self.files_B = sorted(
        #    glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*"))

    def __getitem__(self, index):
        self.A_data = rnaseq  # query rnaseq for biofilm

        self.B_data = rnaseq  # query rnaseq for planktonic

        #image_A = Image.open(self.files_A[index % len(self.files_A)])

        # if self.unaligned:
        #    image_B = Image.open(
        #        self.files_B[random.randint(0, len(self.files_B) - 1)])
        # else:
        #    image_B = Image.open(self.files_B[index % len(self.files_B)])

        # Convert grayscale images to rgb
        # if image_A.mode != "RGB":
        #    image_A = to_rgb(image_A)
        # if image_B.mode != "RGB":
        #    image_B = to_rgb(image_B)

        #item_A = self.transform(image_A)
        #item_B = self.transform(image_B)
        return {"A": A_data, "B": B_data}

    def __len__(self):
        # return max(len(self.files_A), len(self.files_B))
        return max(len(self.A_data), len(self.B_data))
