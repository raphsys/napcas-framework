import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from . import Tensor

def plot_tensor_interactive(tensor, title, output_path):
    """Plots a tensor interactively using Plotly."""
    data = tensor.data()
    fig = go.Figure(data=go.Scatter(y=data, mode='lines+markers'))
    fig.update_layout(title=title, xaxis_title="Index", yaxis_title="Value")
    fig.write_html(output_path)

def plot_training_curves_interactive(losses, accuracies, output_path):
    """Plots training curves interactively using Plotly."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=losses, mode='lines', name='Loss'))
    fig.add_trace(go.Scatter(y=accuracies, mode='lines', name='Accuracy'))
    fig.update_layout(title="Training Curves", xaxis_title="Epoch", yaxis_title="Value", legend=dict(x=0, y=1))
    fig.write_html(output_path)
