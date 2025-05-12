#include "visualization.h"
#include <fstream>
#include <stdexcept>
#include <sstream>

void Visualization::plot_tensor(const Tensor& tensor, const std::string& title, const std::string& output_path) {
    std::ofstream script("plot_tensor.py");
    script << "import matplotlib.pyplot as plt\n";
    script << "import numpy as np\n";
    script << "data = [";
    for (size_t i = 0; i < tensor.data().size(); ++i) {
        script << tensor[i] << (i < tensor.data().size() - 1 ? ", " : "");
    }
    script << "]\n";
    script << "plt.plot(data)\n";
    script << "plt.title('" << title << "')\n";
    script << "plt.savefig('" << output_path << "')\n";
    script << "plt.close()\n";
    script.close();
    system("python plot_tensor.py");
}

void Visualization::log_to_tensorboard(const std::string& metric_name, float value, int step) {
    std::ofstream script("log_tensorboard.py", std::ios::app);
    script << "from torch.utils.tensorboard import SummaryWriter\n";
    script << "writer = SummaryWriter('runs/napcas')\n";
    script << "writer.add_scalar('" << metric_name << "', " << value << ", " << step << ")\n";
    script << "writer.close()\n";
    script.close();
    system("python log_tensorboard.py");
}

void Visualization::plot_training_curves(const std::vector<float>& losses, const std::vector<float>& accuracies, const std::string& output_path) {
    std::ofstream script("plot_curves.py");
    script << "import matplotlib.pyplot as plt\n";
    script << "losses = [";
    for (size_t i = 0; i < losses.size(); ++i) {
        script << losses[i] << (i < losses.size() - 1 ? ", " : "");
    }
    script << "]\n";
    script << "accuracies = [";
    for (size_t i = 0; i < accuracies.size(); ++i) {
        script << accuracies[i] << (i < accuracies.size() - 1 ? ", " : "");
    }
    script << "]\n";
    script << "plt.plot(losses, label='Loss')\n";
    script << "plt.plot(accuracies, label='Accuracy')\n";
    script << "plt.legend()\n";
    script << "plt.savefig('" << output_path << "')\n";
    script << "plt.close()\n";
    script.close();
    system("python plot_curves.py");
}
