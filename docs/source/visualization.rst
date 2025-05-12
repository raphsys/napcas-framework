Visualization
============

The NAPCAS framework provides visualization tools using Matplotlib, TensorBoard, and Plotly.

Matplotlib
----------

Basic plotting of tensors and training curves.

.. code-block:: python

   from napcas import plot_tensor, plot_training_curves

   tensor = Tensor([10], [float(i) for i in range(10)])
   plot_tensor(tensor, "Tensor Plot", "tensor_plot.png")

   losses = [float(i) for i in range(10)]
   accuracies = [0.9 for _ in range(10)]
   plot_training_curves(losses, accuracies, "curves.png")

TensorBoard
-----------

Real-time logging of metrics.

.. code-block:: python

   from napcas import Visualization

   vis = Visualization()
   vis.log_to_tensorboard("Loss", 0.5, step=1)

   # View in TensorBoard
   # tensorboard --logdir runs/napcas

Plotly
------

Interactive visualizations.

.. code-block:: python

   from napcas import plot_tensor_interactive, plot_training_curves_interactive

   tensor = Tensor([10], [float(i) for i in range(10)])
   plot_tensor_interactive(tensor, "Interactive Tensor Plot", "tensor_plot.html")

   losses = [float(i) for i in range(10)]
   accuracies = [0.9 for _ in range(10)]
   plot_training_curves_interactive(losses, accuracies, "curves.html")
