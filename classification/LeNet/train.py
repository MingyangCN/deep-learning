import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

from model import LeNet

def main(batch_size, num_worker=0):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    transform = transforms.Compose(
        [transforms.ToTensor(),                                     # (H, W, C) -> (C, H, W) [0, 1]
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]    # 标准化
    )
    # define training and validation data loaders
    # train: CIFAR-10数据集, 50000张,
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_worker)

    # Test: CIFAR-10数据集, 10000张
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=10000, shuffle=False, num_workers=num_worker)

    test_data_iter = iter(test_loader)
    test_image, test_label = test_data_iter.next()                  # 获得下一批图像

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model = LeNet().to(device=device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)
    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    for epoch in range(10):
        # loop over the dataset multiple times
        running_loss = 0.0

        for step, data in enumerate(train_loader, start=0):
            # get the inputs, data is a list of [inputs, labels]
            inputs = data[0].to(device)
            labels = data[1].to(device)

            # 清零参数的梯度
            optimizer.zero_grad()
            # forward, backward, optimize
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            # 更新权重
            lr_scheduler.step()

            # print
            running_loss += loss.item()                         # 只要值，不要tensor
            if step % 500 == 499:
                with torch.no_grad():                           # 不存储前向传播图的梯度值
                    test_image = test_image.to(device)
                    test_label = test_label.to(device)
                    outputs = model(test_image)
                    predict_y = torch.max(outputs, dim=1)[1]
                    # 两个张量Tensor进行逐元素的比较
                    accuracy = torch.eq(predict_y, test_label).sum().item() / test_label.size(0)

                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0

    print('Finished Training')
    save_pth = './LeNet.pth'
    torch.save(model.state_dict(), save_pth)


if __name__ =="__main__":
    main(batch_size=64, num_worker=2)






