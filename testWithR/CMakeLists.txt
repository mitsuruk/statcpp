# R verification program (standalone, no GTest)
add_executable(verify_vs_r verify_vs_r.cpp)
target_link_libraries(verify_vs_r PRIVATE statcpp)
