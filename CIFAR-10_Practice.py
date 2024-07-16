import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 이미지를 보여주기 위한 함수
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 데이터 로딩 함수
def load_data(batch_size=4, num_workers=2):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

    valset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, valloader, classes

# 모델 정의
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # 배치를 제외한 모든 차원을 평탄화(flatten)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 학습 함수
def train_model(net, trainloader, valloader, criterion, optimizer, num_epochs=10):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        
        train_accuracy = 100 * correct / total
        train_losses.append(running_loss / len(trainloader))
        train_accuracies.append(train_accuracy)
        print(f'Accuracy of the network on the train images after epoch {epoch + 1}: {train_accuracy:.2f}%')

        # 검증 데이터에 대한 평가
        net.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_data in valloader:
                val_inputs, val_labels = val_data
                val_outputs = net(val_inputs)
                val_loss += criterion(val_outputs, val_labels).item()
                _, val_predicted = torch.max(val_outputs, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss / len(valloader))
        val_accuracies.append(val_accuracy)
        print(f'Validation loss after epoch {epoch + 1}: {val_loss / len(valloader):.3f}')
        print(f'Accuracy of the network on the validation images after epoch {epoch + 1}: {val_accuracy:.2f}%')

    print('Finished Training')
    return train_losses, val_losses, train_accuracies, val_accuracies

# 정확도 계산 함수
def calculate_accuracy(net, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# 메인 함수
def main():
    # 데이터 로딩
    trainloader, testloader, valloader, classes = load_data()
    
    # 모델 생성
    net = Net()

    # 손실 함수 및 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 모델 학습
    num_epochs = 10
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(net, trainloader, valloader, criterion, optimizer, num_epochs)

    # 학습 및 검증 손실 시각화
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Train and Validation Accuracy')
    
    plt.show()

    # 모델 저장
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    # 학습 데이터에서 샘플 가져오기
    train_dataiter = iter(trainloader)
    train_images, train_labels = next(train_dataiter)

    # 검증 데이터에서 샘플 가져오기
    test_dataiter = iter(testloader)
    test_images, test_labels = next(test_dataiter)

    # 이미지를 출력합니다.
    imshow(torchvision.utils.make_grid(train_images))
    print('Train GroundTruth: ', ' '.join(f'{classes[train_labels[j]]:5s}' for j in range(4)))

    imshow(torchvision.utils.make_grid(test_images))
    print('Test GroundTruth: ', ' '.join(f'{classes[test_labels[j]]:5s}' for j in range(4)))

    # 모델 로드
    net.load_state_dict(torch.load(PATH))

    # 학습 데이터 예측 수행
    train_outputs = net(train_images)
    _, train_predicted = torch.max(train_outputs, 1)
    print('Train Predicted: ', ' '.join(f'{classes[train_predicted[j]]:5s}' for j in range(4)))

    # 검증 데이터 예측 수행
    test_outputs = net(test_images)
    _, test_predicted = torch.max(test_outputs, 1)
    print('Test Predicted: ', ' '.join(f'{classes[test_predicted[j]]:5s}' for j in range(4)))

    # 모델의 전체 정확도 계산
    train_accuracy = calculate_accuracy(net, trainloader)
    test_accuracy = calculate_accuracy(net, testloader)
    print(f'Accuracy of the network on the train images: {train_accuracy:.2f}%')
    print(f'Accuracy of the network on the test images: {test_accuracy:.2f}%')

if __name__ == '__main__':
    main()
