import torch
from torchvision import models
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data_utils
from LoadData import ReadData_fer

EPOCH = 25

def train_testLoader():
    # Train and Test data in numpy format
    x_train, y_train, x_test, y_test = ReadData_fer()

    # Train Loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train,batch_size=100,shuffle=True)

    # Test Loader
    test = data_utils.TensorDataset(torch.from_numpy(x_test),torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test,batch_size = 100,shuffle=True)

    # print(train_loader)
    #
    # i = 0
    # for bacth,label in train_loader:
    #     print("i: ",i)
    #     print(bacth.size())
    #     print(label.size())
    #     i = i + 1
    return train_loader,test_loader


def resnet18_fine_tune():
    model_ft = models.resnet18(pretrained=True)

    for param in model_ft.parameters():
        param.requires_grad = True

    fc_features = model_ft.fc.in_features # Extract the number of the fully connected layer
    model_ft.fc = nn.Linear(fc_features,7)

    return model_ft

def Train_model(train_loader,test_loader):

    # Load model, criterion, optimizer (model built-in optimizer)
    model = resnet18_fine_tune()
    criterion = nn.CrossEntropyLoss
    optimizer = torch.optim.SGD(model.parameters(),lr=0.001)


    best_model = model
    best_acc = 0

    for epoch in range(EPOCH):

        running_loss = 0.0
        running_corrects = 0

        for image, label in train_loader:
            image = Variable(image)
            label = Variable(label)

            # Forward and backward
            outputs = model(image)
            _, predict = torch.max(outputs.data, 1)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            # collect data information
            running_loss += loss.data[0]
            running_corrects += torch.sum(predict == label.data)

    model.eval()

    correct = 0
    total = 0

    for image,label in test_loader:
        image = Variable(image)
        outputs = model(image)
        _,predict = torch.max(outputs.data,1)
        total += label.size(0)
        correct += (predict.cpu() == label).sum()

    print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))


train_loader, test_loader = train_testLoader()
Train_model(test_loader,test_loader)
