Usage
=====

Python Example
--------------

.. code-block:: python

   from napcas import NAPCAS, DataLoader, SGD, CrossEntropyLoss

   loader = DataLoader("data/synthetic.csv", batch_size=2)
   model = NAPCAS(10, 5)
   optimizer = SGD([model], lr=0.01)
   loss_fn = CrossEntropyLoss()

   input, target = loader.next()
   output = Tensor([2, 5])
   model.forward(input, output)
   loss = loss_fn.forward(output, target)
   grad_output = loss_fn.backward(output, target)
   grad_input = Tensor(input.shape())
   model.backward(grad_output, grad_input)
   optimizer.step()
