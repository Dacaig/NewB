import syft as sy
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]='True'
torch.manual_seed(1)


class augments():
    def __init__(self):
        self.epoches=2
        self.batch_size=50
        self.test_batch_size=50
        self.learning_rate=0.001



train_data=torchvision.datasets.MNIST(root="./mnist",train=True,transform=torchvision.transforms.ToTensor(),download=True)

print(train_data.data.size())
plt.imshow(train_data.data[0].numpy())
#plt.show()

test_data=torchvision.datasets.MNIST(root="./mnist",train=False)
print(test_data.data.size())

class CNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1=nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.output=nn.Linear(in_features=32*7*7,out_features=10)

    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(x.size(0),-1)
        output=self.output(x)
        return output

def main():
    cnn=CNN()
    arg=augments()
    optimizer=torch.optim.Adam(cnn.parameters(),lr=arg.learning_rate)
    loss_function=nn.CrossEntropyLoss()

    for epoch in range(arg.epoches):
        print("in{}epoch".format(epoch))
        for step,(batch_x,batch_y) in enumerate(train_data):
            output=cnn(batch_x)
            loss=loss_function(output,batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 ==0:
                test_output=cnn(test_x)
                pred_y=torch.max(test_output,1)[1].data.numpy()
                accuracy=((pred_y==test_y.data.numpy()).astype(int).sum())/float(test_y.size(0))
                print('Epoch:',epoch,'|train loss:%.4f' %loss.data.numpy(),'|test accuracy:%.2f'% accuracy)
    test_output=cnn(text_x[:10])
    pred_y=torch.max(test_output,1)[1].data.numpy().squeeze()
    print(pred_y)

if __name__=="__main__":
    main()