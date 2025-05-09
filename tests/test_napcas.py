import napcas

# Exemple d'utilisation
x = napcas.Tensor([1, 10], [0.5]*10)
y = napcas.Tensor([1, 5], [0.0]*5)

cell = napcas.NNCel(10, 5)
cell.forward(x)
cell.backward(y)
cell.update(0.01)
