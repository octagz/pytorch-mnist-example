import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Define una red neuronal simple
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # Carga el modelo entrenado desde el archivo
    model = SimpleNet()
    model.load_state_dict(torch.load("modelo_entrenado_simple.pth"))
    model.eval()


    # Check if an argument is provided
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <image_path>")
        sys.exit(1)

    # Carga una imagen de entrada y redimensiona
    image_path = sys.argv[1]  # Get the image path from command line argument

    transform = transforms.Compose([transforms.Resize((28, 28)),  # Redimensiona a 28x28
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
