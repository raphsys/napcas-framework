# Exécutable de test unitaire Tensor
add_executable(test_tensor cpp/test_tensor.cpp)
target_include_directories(test_tensor PRIVATE cpp/include)
target_link_libraries(test_tensor PRIVATE napcas_core)

