#Experiment 1
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
import time
import torch.nn.utils.prune as prune
import torch.cuda.amp as amp

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.Grayscale(num_output_channels=3),  # 转换为具有三个通道的灰度图像
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化
])

# 加载 MNIST 数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义未修改的 ResNet 模型
class OriginalResNet(nn.Module):
    def __init__(self):
        super(OriginalResNet, self).__init__()
        self.resnet = resnet18(pretrained=False, num_classes=10)

    def forward(self, x):
        return self.resnet(x)

# 定义预训练的 ResNet 模型
class PretrainedResNet(nn.Module):
    def __init__(self):
        super(PretrainedResNet, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)

    def forward(self, x):
        return self.resnet(x)

# 定义参数共享的 ResNet 模型
class SharedResNet(nn.Module):
    def __init__(self):
        super(SharedResNet, self).__init__()
        # 共享的卷积层
        self.shared_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 修改输入通道数为3
        # 分支 1
        self.branch1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # 分支 2
        self.branch2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # 分支 3
        self.branch3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # 最后的全连接层
        self.fc = nn.Linear(64 * 3, 10)

    def forward(self, x):
        # 共享的卷积层
        x = self.shared_conv1(x)
        # 分支 1
        out1 = self.branch1(x)
        out1 = nn.functional.adaptive_avg_pool2d(out1, (1, 1))
        out1 = torch.flatten(out1, 1)
        # 分支 2
        out2 = self.branch2(x)
        out2 = nn.functional.adaptive_avg_pool2d(out2, (1, 1))
        out2 = torch.flatten(out2, 1)
        # 分支 3
        out3 = self.branch3(x)
        out3 = nn.functional.adaptive_avg_pool2d(out3, (1, 1))
        out3 = torch.flatten(out3, 1)
        # 合并三个分支
        x = torch.cat((out1, out2, out3), dim=1)
        # 最后的全连接层
        x = self.fc(x)
        return x

# 混合精度训练和评估模型的函数
def train_and_evaluate_model_mixed_precision(model, optimizer, train_loader, test_loader, epochs, criterion):
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if batch_idx % 100 == 0:
                print(
                    f'Epoch {epoch}: [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    test_loss = 0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            total += target.size(0)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {100. * correct / total:.0f}%')
# 训练和评估模型的函数
def train_and_evaluate_model(model, optimizer, train_loader, test_loader, epochs, criterion):
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(
                    f'Epoch {epoch}: [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    test_loss = 0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            total += target.size(0)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {100. * correct / total:.0f}%')

    
# 第一组对比实验：原始的 ResNet 模型
print("Training and evaluating the Original ResNet model:")
original_resnet_model = OriginalResNet().to(device)
optimizer_original_resnet = optim.SGD(original_resnet_model.parameters(), lr=0.001, momentum=0.9)
train_and_evaluate_model_mixed_precision(original_resnet_model, optimizer_original_resnet, train_loader, test_loader, epochs=1, criterion=criterion)

# 第二组对比实验：预训练的 ResNet 模型
print("Training and evaluating the Pretrained ResNet model:")
pretrained_resnet_model = PretrainedResNet().to(device)
optimizer_pretrained_resnet = optim.SGD(pretrained_resnet_model.parameters(), lr=0.001, momentum=0.9)
train_and_evaluate_model(pretrained_resnet_model, optimizer_pretrained_resnet, train_loader, test_loader, epochs=3, criterion=criterion)

# 第三组对比实验：参数共享的 ResNet 模型
print("Training and evaluating the Shared ResNet model:")
shared_resnet_model = SharedResNet().to(device)
optimizer_shared_resnet = optim.SGD(shared_resnet_model.parameters(), lr=0.001, momentum=0.9)
train_and_evaluate_model(shared_resnet_model, optimizer_shared_resnet, train_loader, test_loader, epochs=3, criterion=criterion)

# 第四组对比实验：剪枝修改后的 ResNet 模型
print("Training and evaluating the Pruned ResNet model:")
pruned_resnet_model = OriginalResNet().to(device)
pruned_resnet_model = prune_model(pruned_resnet_model)
optimizer_pruned_resnet = optim.SGD(pruned_resnet_model.parameters(), lr=0.001, momentum=0.9)
train_and_evaluate_model(pruned_resnet_model, optimizer_pruned_resnet, train_loader, test_loader, epochs=3, criterion=criterion)

# 第五组对比实验：使用混合精度训练的 ResNet 模型
print("Training and evaluating the Mixed Precision ResNet model:")
mixed_precision_resnet_model = OriginalResNet().to(device)
optimizer_mixed_precision_resnet = optim.SGD(mixed_precision_resnet_model.parameters(), lr=0.001, momentum=0.9)
train_and_evaluate_model_mixed_precision(mixed_precision_resnet_model, optimizer_mixed_precision_resnet, train_loader, test_loader, epochs=3, criterion=criterion)
