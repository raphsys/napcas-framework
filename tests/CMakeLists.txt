# CMakeLists.txt pour le répertoire tests

# Trouver Google Test
find_package(GTest REQUIRED)

# Ajouter les tests unitaires
add_executable(test_napcas test_all_modules.cpp test_napcas.cpp)
target_link_libraries(test_napcas GTest::GTest GTest::Main libnapcas)

# Configurer les tests avec CTest
enable_testing()
add_test(NAME test_napcas COMMAND test_napcas)
