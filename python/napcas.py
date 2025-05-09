import os
import sys

# Ajouter le chemin du module C++
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cpp', 'build'))

try:
    import napcas
except ImportError as e:
    print(f"Erreur d'importation du module C++: {e}")
    print("Assurez-vous que le module a été correctement compilé.")
    raise

# Exposer les classes dans le module napcas
from napcas import Linear, Conv2d, ReLU, Sigmoid, Tanh, MSELoss, CrossEntropyLoss, SGD, Adam, DataLoader, Autograd, NAPCAS

# Fonction pour créer un réseau simple
def create_model():
    return [
        Linear(784, 128),
        ReLU(),
        Linear(128, 10)
    ]

# Fonction d'entraînement
def train(model, dataloader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        total_loss = 0.0
        batch_count = 0
        for inputs, targets in dataloader.next():  # Correction: itérer sur les batches
            # Forward pass
            output = inputs
            for layer in model:
                temp = napcas.Tensor(inputs.shape())  # Tenseur temporaire pour stocker la sortie
                layer.forward(output, temp)
                output = temp

            # Calcul de la perte
            loss = criterion.forward(output, targets)
            grad = criterion.backward(output, targets)

            # Backward pass
            grad_input = grad
            for layer in reversed(model):
                temp = napcas.Tensor(inputs.shape())  # Tenseur temporaire pour le gradient
                layer.backward(grad_input, temp)
                grad_input = temp

            # Mise à jour des paramètres
            optimizer.step()

            total_loss += loss
            batch_count += 1

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / batch_count:.4f}")

# Fonction d'évaluation
def evaluate(model, dataloader):
    correct = 0
    total = 0
    for inputs, targets in dataloader.next():
        # Forward pass
        output = inputs
        for layer in model:
            temp = napcas.Tensor(inputs.shape())
            layer.forward(output, temp)
            output = temp

        # Prédictions
        predicted = output.data().argmax()
        label = targets.data().argmax()

        if predicted == label:
            correct += 1
        total += 1

    print(f"Précision: {correct / total * 100:.2f}%")
