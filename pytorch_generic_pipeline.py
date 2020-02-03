import cv2
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from torch.utils.data import DataLoader, Dataset

############################### DATASET #######################################

################## PREPARATION #####################
class Custom_Dataset(Dataset):
    """ Custom Dataset Loading and using """

    def __init__(self, root_dir, transform = None, csv_file = None):
        """
            Args: ......
        """
        self.csv_file = pd.read_csv(csv_file) ## if csv present
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """ returns the size of the dataset """
        return len(self.csv_file)

    def __getitem__(self, idx):
        """ support indexing to get ith sample from the dataset """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = IMAGE_NAME ## GET THE IMAGE NAME
        image = cv2.imread(img_name)
        # use CSV file if needed
        # if need to return both image and some other details
        # use dictionary
        # sample = {"image": image, ....}

        if self.transform:
            image = self.transform(image) # image/sample

        return image # / sample


""" Tranforms ( Tranforming the image to proper needs ) """



# Now a lot of times in real world datasets the images are not uniform
# and hence applying transform individually might become painful, so
# declare class for that

class Rescale(object):
    """ Rescale Image to a given size """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = image

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h>w:
                new_h, new_w = self.output_size*h/w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size*w/h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = cv2.resize(image, (new_h, new_w))

        # Change CSV file accordingly
        return image # / dictionary of image and other data


class ToTensor(object):
    """ Convert PIL Image to tensors """
    def __call__(self, image):
        image = image.transpose((2, 0, 1)) # convert to torch required format
                                           # (C, H, W)
        return torch.from_numpy(image)


"""
    Now that we have defined the transform operations
    compose them together
"""

scale = Rescale(256)
composed = transforms.Compose([Rescale(256)])

# Applying on the dataset
transformed_dataset = Custom_Dataset(root_dir, transform=transforms.Compose([
                                                            Rescale(256),
                                                            ToTensor()
                                                        ]))

# For more info look at torchvision.ImageFolder which makes it easier to
# get the job done if data is structured and dir names = class names
# to split the dataset after the dataset is ready and before batching and
# shuffling you can use torch.utils.data.random_split() to split into test
# set and val set

############# BATCHING, SHUFFLING #############
dataloader = DataLoader(transformed_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                                        num_workers=-1)





############################### MODEL #######################################
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.Layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.Layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(7*7*64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.Layer1(x)
        out = self.Layer2(x)
        out = out.reshape(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


model = Net()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

#Training model
total_step = len(train_loader)
loss_list, acc_list = [], []

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        #Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct/total)


# when testing
model.eval() ############ Most Important
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


# Storing the model
torch.save(model.state_dict(), MODEL_STORE_PATH + 'model.ckpt')
