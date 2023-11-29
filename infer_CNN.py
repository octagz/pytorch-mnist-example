import torch
import torch.nn as nn
import torchvision.transforms as transforms
import sys
from PIL import Image

# Define la misma arquitectura de red neuronal convolucional (CNN) utilizada en el entrenamiento
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

if __name__ == "__main__":
    # Carga el modelo entrenado desde el archivo
    model = CNN()
    model.load_state_dict(torch.load("modelo_entrenado_cnn.pth"))
    model.eval()

    # Check if an argument is provided
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <image_path>")
        sys.exit(1)

    # Carga una imagen de entrada y redimensiona
    image_path = sys.argv[1]  # Get the image path from command line argument

    transform = transforms.Compose([transforms.Grayscale(),
                                    transforms.Resize((28, 28)),  # Redimensiona a 28x28
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.5,), (0.5,))])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)

    # Realiza una inferencia
    output = model(image)

    # Obtén las probabilidades de clasificación
    probabilities = torch.softmax(output, dim=1)

    # Obtén los índices de los tres números más probables
    top3_indices = torch.topk(probabilities, 3, dim=1)[1][0]

    # Define una lista de etiquetas para los dígitos
    digit_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # Imprime los tres números más probables
    print("Los tres números más probables son:")
    for idx in top3_indices:
        digit = digit_labels[idx]
        prob = probabilities[0, idx].item()
        print(f"Número: {digit}, Probabilidad: {prob:.4f}")
