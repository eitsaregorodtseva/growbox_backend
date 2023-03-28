import numpy as np
import torch
import pickle
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import io
# import PIL
from io import BytesIO

from PIL import Image


DATA_MODES = ['train', 'val', 'test']
RESCALE_SIZE = 224
DEVICE = torch.device("cpu")

# Очень простая сеть
class SimpleCnn(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.out = nn.Linear(96 * 5 * 5, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.view(x.size(0), -1)
        logits = self.out(x)
        return logits


def predict_one_sample(model, inputs):
    """Предсказание, для одной картинки"""
    with torch.no_grad():
        inputs = inputs.to(DEVICE)
        model.eval()
        logit = model(inputs).cpu()
        print(logit)

        probs = torch.nn.functional.softmax(logit, dim=-1).numpy()
    return probs


def prepareFile(file):
    # print(file)
    # tf = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize((512, 640)),
    #     transforms.ToTensor()
    # ])
    # test_file = [Path("/content/pic_0037.jpg")]
    # test_file = GrowBoxDataset(file, mode='test')

    # test_file = tf(file)
    # print(test_file)

    # image = Image.open(file)
    # image.load()
    file = file.resize((RESCALE_SIZE, RESCALE_SIZE))
    file = np.array(file)
    file = np.array(file / 255, dtype='float32')
    print(file.shape)

    convert_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    test_file = convert_tensor(file)
    # print(test_file)
    print(test_file)
    #
    simple_cnn = SimpleCnn(3).to('cpu')
    simple_cnn.load_state_dict(torch.load('model_weights.pth', map_location='cpu'))
    simple_cnn.eval()
    # #
    probs_im = predict_one_sample(simple_cnn, test_file.unsqueeze(0))
    print(probs_im)
    #
    label_encoder = pickle.load(open("label_encoder.pkl", 'rb'))
    #
    y_pred = np.argmax(probs_im, -1)
    print(probs_im)
    print(label_encoder.classes_[y_pred][0])
    # return 1
    return label_encoder.classes_[y_pred][0]


def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    image.load()
    # tf = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    # convert_tensor = transforms.ToTensor()

    # tens = tf(image)
    # print(tens)
    return image
