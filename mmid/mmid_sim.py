import glob
import os
from optparse import OptionParser

import torch
from PIL import Image
from apex import amp
from torch.utils.data import Dataset
from torchvision import models
from torchvision import transforms


def get_option_parser():
    parser = OptionParser()
    parser.add_option("--e", dest="en_folder", metavar="FILE", default=None)
    parser.add_option("--f", dest="foreign_folder", metavar="FILE", default=None)
    parser.add_option("--o", dest="output_file", metavar="FILE", default=None)
    parser.add_option("--fp16", action="store_true", dest="fp16", default=False)
    parser.add_option("--small", action="store_true", dest="small_resnet", default=False)
    return parser


class ModifiedResnet(models.ResNet):
    def _forward_impl(self, x):
        input = x
        x1 = self.conv1(input)
        x2 = self.bn1(x1)
        x3 = self.relu(x2)
        x4 = self.maxpool(x3)

        x5 = self.layer1(x4)
        x6 = self.layer2(x5)
        x7 = self.layer3(x6)
        x8 = self.layer4(x7)

        x9 = self.avgpool(x8)
        x10 = torch.flatten(x9, 1)
        x11 = self.fc(x10)

        grid_hidden = x8.view(x8.size(0), -1)

        return torch.cat([x11, grid_hidden], dim=1)


def init_net(small):
    if small:
        model = models.resnet18(pretrained=True)
    else:
        model = models.resnet152(pretrained=True)
    model.__class__ = ModifiedResnet
    model.eval()

    return model


class MMID(Dataset):
    def __init__(self, image_dir: str):
        self.images = glob.glob(os.path.join(image_dir, "*.jpg"))
        self.size_transform = transforms.Resize(256)
        self.crop = transforms.CenterCrop(224)
        self.to_tensor = transforms.ToTensor()
        self.img_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.images)

    def get_img(self, path):
        try:
            with Image.open(path) as im:
                # make sure not to deal with rgba or grayscale images.
                img = im.convert("RGB")
                img = self.crop(self.size_transform(img))
                im.close()
        except:
            print("Corrupted image", path)
            img = Image.new('RGB', (224, 224))
        return img

    def __getitem__(self, item):
        return torch.stack(list(map(lambda item: self.img_normalize(self.to_tensor(self.get_img(item))), self.images)))


if __name__ == "__main__":
    parser = get_option_parser()
    (options, args) = parser.parse_args()
    image_model = init_net(options.small_resnet)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_model = image_model.to(device)
    if options.fp16 and torch.cuda.is_available():
        image_model = amp.initialize(image_model, opt_level="O2")

    foreign_image_dir = options.foreign_folder
    english_image_dir = options.en_folder
    pin_memory = torch.cuda.is_available()

    en_folders, en_vectors = [], []

    for en_folder in os.listdir(english_image_dir):
        en_im_folder = os.path.join(english_image_dir, en_folder)
        for en_im_dir in os.listdir(en_im_folder):
            en_dir = os.path.join(en_im_folder, en_im_dir)
            en_mmid_data = MMID(en_dir)
            data = en_mmid_data[0].to(device)
            with torch.no_grad():
                vector = image_model(data)
                vnorm = torch.norm(vector, dim=-1, p=2).unsqueeze(-1)
                vector = torch.div(vector, vnorm)
                en_vectors.append(vector.cpu())
                en_folders.append(en_dir)

            print(en_dir, len(en_vectors))

    foreign_folders, foreign_vectors = [], []
    for foreign_folder in os.listdir(foreign_image_dir):
        f_dir = os.path.join(foreign_image_dir, foreign_folder)
        foreign_mmid_data = MMID(f_dir)
        data = foreign_mmid_data[0].to(device)
        with torch.no_grad():
            vector = image_model(data)
            vnorm = torch.norm(vector, dim=-1, p=2).unsqueeze(-1)
            vector = torch.div(vector, vnorm)
            foreign_vectors.append(vector.cpu())
            foreign_folders.append(f_dir)

        print(f_dir, len(foreign_vectors))



    with torch.no_grad(), open(options.output_file, "w") as writer:
        for f_folder, f_vector in zip(foreign_folders, foreign_vectors):
            for en_folder, en_vector in zip(en_folders, en_vectors):
                fv = f_vector.to(device)
                ev = en_vector.to(device)
                cosines = torch.mm(fv, ev.T)
                max_f = torch.max(cosines, dim=-1)[0].squeeze()
                avg_max_sim = float((torch.sum(max_f) / len(fv)).cpu())

                writer.write("\t".join([f_folder, en_folder, str(avg_max_sim)]))
                writer.write("\n")
                print(f_folder, en_folder, avg_max_sim)

                f_vector.cpu()
                en_vector.cpu()
