# for classification

#import generic libraries
import os
import cv2
import copy
import time
import torch
import torchvision
import torch.nn as nn
from torchvision import models, datasets, transforms

# dictionary of transforms to be applied on train and validation datasets
# respectively
data_transform = {
    "train": transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(224),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # must
        # and can add bunch of more
    ]),
    "val" : transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(224),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # must
        transforms.Resize(256) # must
        # and can add bunch of more
    ])
}

root_dir = "" #specifuy root directory
# if images being loaded from a structured directory of the format
# root_dir/
#          class1/
#                   1.jpg
#                   2.jpg
#                   3.jpg....
#          class2/
#                   1.jpg
#                   2.jpg
#                   3.jpg....
# ......
# else use the method shown in generic pipeline file
image_datsets = {x: datasets.ImageFolder(path, data_transform[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datsets[x], batch_size, shuffle =True, num_workers=-1) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datsets[x]) for x in ['train', 'val']}
class_names = image_datsets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


####################### MODEL ###########################

def train_model(model, criterion, optimizer, scheduler, num_epochs):
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in num_epochs:
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_correct = 0

            for images, labels in dataloaders[phase]:
                image = image.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward
                # track history if in train_model
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(image)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()
                running_corrects += torch.sum(preds==labels.data)

            if phase == 'train':
                scheduler.step()

            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model


#### Loading a model #####
model = torchvision.models.inception_v3(pretrained=True)

# if finetuning need to be done then follow this section else skip and go to
# next section

""" FINETUNING """
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names)) # fully connected layer from
                                                    # last of trained model to last which has no of nodes = num_classes
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) ## what is this ???

""" FEATURE EXTRACTOR RATHER THAN FINETUNING """
for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


############# TRAIN AND EVALUATE #################
model = train(mode, criterion, optimizer, scheduler, 25)

###### PREDICT ######
model.eval()
outputs = model(image)
_, preds = torch.max(outputs, 1)
print(class_names[preds])
