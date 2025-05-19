#include <iostream>
#include "napcas/tensor.h"

using namespace napcas;

int main() {
    Tensor a = Tensor::ones({2, 2});
    Tensor b = Tensor::ones({2, 2});
    Tensor c = a + b;

    std::cout << "Résultat :\\n";
    c.print_summary();
    return 0;
}
