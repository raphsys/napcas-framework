import os
import sys

# Ajouter le chemin du module C++ (si nécessaire)
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
        for inputs, targets in dataloader:
            outputs = model[0].forward(inputs)
            outputs = model[1].forward(outputs)
            outputs = model[2].forward(outputs)

            loss = criterion.forward(outputs, targets)
            grad = criterion.backward(outputs, targets)

            model[2].backward(grad)
            model[1].backward(grad)
            model[0].backward(grad)

            optimizer.step()

            total_loss += loss

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")

# Fonction d'évaluation
def evaluate(model, dataloader):
    correct = 0
    total = 0
    for inputs, targets in dataloader:
        outputs = model[0].forward(inputs)
        outputs = model[1].forward(outputs)
        outputs = model[2].forward(outputs)

        predicted = outputs.data.argmax()
        label = targets.data.argmax()

        if predicted == label:
            correct += 1
        total += 1

    print(f"Précision: {correct / total * 100:.2f}%")
