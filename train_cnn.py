import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Verifica si CUDA está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define una red neuronal convolucional (CNN) simple
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)  # 1 canal de entrada, 32 mapas de características, tamaño de kernel 3x3
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)  # Capa de Max Pooling
        self.conv2 = nn.Conv2d(32, 64, 3)  # 32 mapas de características de entrada, 64 mapas de características de salida, tamaño de kernel 3x3
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2)  # Capa de Max Pooling
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # Capa completamente conectada
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)  # Capa de salida
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.reshape(x.size(0), -1)  # Redimensiona los datos
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

# Carga y transforma datos MNIST
transform = transforms.Compose([transforms.Grayscale(),  # Convierte las imágenes a escala de grises
                                transforms.Resize((28, 28)),  # Redimensiona a 28x28
                                transforms.ToTensor(), 
                                transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Crea una instancia del modelo y envíalo a la GPU
net = CNN().to(device)

# Define la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Entrena el modelo en la GPU
for epoch in range(10):  # Ejemplo: 10 épocas de entrenamiento
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Época {epoch + 1}, Pérdida: {running_loss / (i + 1)}")

print("Entrenamiento completo")

# Guarda el modelo entrenado en un archivo
torch.save(net.state_dict(), "modelo_entrenado_cnn.pth")
