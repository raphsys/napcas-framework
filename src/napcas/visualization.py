from . import Visualization

def plot_tensor(tensor, title, output_path):
    Visualization.plot_tensor(tensor, title, output_path)

def log_to_tensorboard(metric_name, value, step):
    Visualization.log_to_tensorboard(metric_name, value, step)

def plot_training_curves(losses, accuracies, output_path):
    Visualization.plot_training_curves(losses, accuracies, output_path)
