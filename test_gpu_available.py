import torch

# Verifica si CUDA está disponible
if torch.cuda.is_available():
    # Imprime el nombre de la GPU disponible
    print("GPU disponible:", torch.cuda.get_device_name(0))
else:
    print("No se encontró una GPU disponible.")
