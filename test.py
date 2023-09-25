import timm
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from glob import glob
import pandas as pd
from torchvision.transforms import ToTensor
from PIL import Image
import sys
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 4
image_count = 25
tf = ToTensor()
transition_path = sys.argv[1]
not_transition_path = sys.argv[2]
csv_path = sys.argv[3]

image_transition_path = transition_path
image_not_transition_path = not_transition_path
csv = pd.read_csv(csv_path, encoding='cp949')


class Custom_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = timm.create_model('resnet34', pretrained=True)
        self.hidden_size = 128
        self.num_layers = 4
        self.rnn = nn.RNN(1000, 128, 4, batch_first=True)
        self.fc = nn.Linear(128, 2)
        self.cfs = nn.Linear(2, 1000)
        self.cfa = nn.Linear(1, 1000)
        self.fc1 = nn.Linear(1000, 1)
        self.fc5 = nn.Linear(4, 2)

    def forward(self, inputs, sex, age):
        total_x = torch.empty((batch_size, image_count, 1)).to(device)
        final_x = torch.empty((batch_size, 1)).to(device)
        for i in range(len(inputs)):
            for j in range(len(inputs[i])):
                x = self.cnn1(inputs[i, j].to(device))
                x = self.fc1(x).to(device)
                total_x[i, j] = x.to(device)
            total_x[i], indices = torch.sort(total_x[i], dim=1)
            final_x[i] = total_x[i, -1].to(device)

        x = torch.cat([final_x, sex], dim=1).to(device)
        x = torch.cat([x, age], dim=1).to(device)
        x = self.fc5(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, image_list, sex, age, label):
        self.img_path = image_list

        self.label = label
        self.sex = sex
        self.age = age

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        image_tensor = self.img_path[idx]
        clinical_tensor = self.sex[idx]
        clinical_age_tensor = self.age[idx]
        label_tensor = self.label[idx]
        return (image_tensor, clinical_tensor, clinical_age_tensor), label_tensor


image_list = []
label_list = []
image_transition_list = glob(image_transition_path)
image_transition_label = torch.ones(len(image_transition_list), 2)
image_not_transition_list = glob(image_not_transition_path)
image_not_transition_label = torch.zeros(len(image_not_transition_list), 2)
image_list.extend(image_transition_list)
image_list.extend(image_not_transition_list)
label_list.extend(image_transition_label)
label_list.extend(image_not_transition_label)


image_5x_tensor = torch.empty((len(image_list), image_count, 1, 3, 256, 256))


clinical_tensor = torch.zeros((len(image_list), 2))
clinical_age_tensor = torch.zeros((len(image_list), 1))
for i in range(len(image_list)):
    image_file_list = glob(image_list[i]+'/*.jpg')
    image_index = torch.randint(low=0, high=len(
        image_file_list)-1, size=(image_count,))
    count = 0
    for index in image_index:
        image = 1 - \
            tf(Image.open(image_file_list[index]).resize(
                (256, 256))).unsqueeze(0)
        image_5x_tensor[i, count] = image.unsqueeze(0)
        count += 1

    clincal_feature = csv.loc[csv['데이터톤번호'] ==
                              os.path.basename(image_list[i])].reset_index()
    clinical_age_tensor[i, 0] = clincal_feature.loc[0]['나이']/100
    if clincal_feature.loc[0]['성별'] == 'Female':
        clinical_tensor[i, 1] = 1
    else:
        clinical_tensor[i, 0] = 1


dataset = CustomDataset(image_5x_tensor, clinical_tensor,
                        clinical_age_tensor, label_list)


dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, drop_last=True)

model = Custom_model()
model = model.to(device)
model.load_state_dict(torch.load(
    '../../model/image_5x/urbs_final.pt'))
with torch.no_grad():
    total_y = torch.zeros((len(dataloader), 2)).to(device)
    total_prob = torch.zeros((len(dataloader), 2)).to(device)
    count = 0
    model.eval()
    for x, y in dataloader:
        y = y.to(device).float()
        x[0] = x[0].to(device).float()
        x[1] = x[1].to(device).float()
        x[2] = x[2].to(device).float()

        predict = model(x[0], x[1], x[2]).to(device)
        cost = F.cross_entropy(predict.softmax(dim=1), y)
        prob_pred = predict.softmax(dim=1)
        total_y[count] = y
        total_prob[count] = prob_pred
        count += 1
print(total_prob.cpu())
