#Experiment 2
#使用MNIST数据集
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

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义未修改的 ResNet 模型
class OriginalResNet(nn.Module):
    def __init__(self):
        super(OriginalResNet, self).__init__()
        self.resnet = resnet18(pretrained=False, num_classes=10)

    def forward(self, x):
        return self.resnet(x)

# 定义剪枝修改后的 ResNet 模型
def prune_model(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.2)
    return model

def train_and_evaluate_model_mixed_precision(model, optimizer, train_loader, test_loader, epochs, criterion):
    scaler = torch.cuda.amp.GradScaler()
    losses = []
    accuracies = []

    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0

        # 训练模型
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
            epoch_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            total += target.size(0)
            correct += pred.eq(target.view_as(pred)).sum().item()

        epoch_loss /= len(train_loader.dataset)
        accuracy = 100. * correct / total
        losses.append(epoch_loss)
        accuracies.append(accuracy)
        print(f'Epoch {epoch}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')

    # 测试模型
    test_loss = 0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        start_time = time.time()  # 记录推理开始时间
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            total += target.size(0)
            correct += pred.eq(target.view_as(pred)).sum().item()
        end_time = time.time()  # 记录推理结束时间
        inference_time = end_time - start_time  # 计算推理时间

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / total
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {test_accuracy:.0f}%, Inference time: {inference_time:.4f} seconds')
    return losses, accuracies


# 训练和评估原始的 ResNet 模型
print("Training and evaluating the Original ResNet model:")
original_resnet_model = OriginalResNet().to(device)
optimizer_original_resnet = optim.SGD(original_resnet_model.parameters(), lr=0.001, momentum=0.9)
train_and_evaluate_model_mixed_precision(original_resnet_model, optimizer_original_resnet, train_loader, test_loader, epochs=3, criterion=nn.CrossEntropyLoss())

# 训练和评估剪枝和混合精度训练的 ResNet 模型
print("Training and evaluating the Pruned Mixed Precision ResNet model:")
mixed_precision_pruned_resnet_model = OriginalResNet().to(device)
mixed_precision_pruned_resnet_model = prune_model(mixed_precision_pruned_resnet_model)
optimizer_mixed_precision_pruned_resnet = optim.SGD(mixed_precision_pruned_resnet_model.parameters(), lr=0.001, momentum=0.9)
train_and_evaluate_model_mixed_precision(mixed_precision_pruned_resnet_model, optimizer_mixed_precision_pruned_resnet, train_loader, test_loader, epochs=3, criterion=nn.CrossEntropyLoss())
