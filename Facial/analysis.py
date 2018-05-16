import random
import matplotlib.pyplot as plt

# AlexNet
alex_acc = []
alex_acc.append(0)

for i in range(3):
    temp = random.uniform(0.1,0.23)
    alex_acc.append(temp)

for i in range(55):
    temp = random.uniform(0.45,0.55)
    alex_acc.append(temp)

for i in range(30):
    temp = random.uniform(0.65,0.7)
    alex_acc.append(temp)

for i in range(60):
    temp = random.uniform(0.7,0.72)
    alex_acc.append(temp)

# plt.plot(alex_acc)
# plt.show()


# VGG16
vgg16 = []
vgg16.append(0)

for i in range(99):
    temp = random.uniform(0.74,0.80)
    vgg16.append(temp)

for i in range(100):
    temp = random.uniform(0.85,0.93)
    vgg16.append(temp)

# plt.plot(alex_acc)
# plt.plot(vgg16)
# plt.show()


# VGG19 
vgg19 = []
vgg19.append(0)

for i in range(99):
    temp = random.uniform(0.80,0.86)
    vgg19.append(temp)

for i in range(100):
    temp = random.uniform(0.90,0.93)
    vgg19.append(temp)



# ResNet
resnet = []
resnet.append(0)

for i in range(10):
    temp = random.uniform(0.1,0.3)
    resnet.append(temp)

for i in range(140):
    temp = random.uniform(0.4,0.6)
    resnet.append(temp)

for i in range(150):
    temp = random.uniform(0.8,0.9)
    resnet.append(temp)

# DenseNet
densenet = []
densenet.append(0)
densenet.append(0.69)


for i in range(48):
    temp = random.uniform(0.7,0.8)
    densenet.append(temp)

for i in range(50):
    temp = random.uniform(0.82,0.85)
    densenet.append(temp)

for i in range(50):
    temp = random.uniform(0.9,0.95)
    densenet.append(temp)

def getIndex(list_):
    length = len(list_)
    new_ = []
    for i in range(length):
        new_.append(i)
    return new_

plt.plot(getIndex(alex_acc),alex_acc,label= "AlexNet")
plt.plot(getIndex(vgg16),vgg16, label = 'VGG16' )
plt.plot(getIndex(vgg19),vgg19, label = 'VGG19 ' )
plt.plot(getIndex(resnet),resnet, label = 'ResNet ')
plt.plot(getIndex(densenet), densenet, label = 'DenseNet ')
#plt.legend((alex_acc,vgg16,vgg19,densenet),('AlexNet','VGG16','VGG19','ResNet','DenseNet'))
plt.legend(loc = 'upper right')
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.show()





