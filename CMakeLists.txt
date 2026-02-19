cmake_minimum_required(VERSION 3.14)
project(statcpp
    VERSION 0.1.0
    LANGUAGES CXX
    DESCRIPTION "C++17 Header-only Statistics Library"
)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Language option for Japanese commented headers
option(STATCPP_USE_JAPANESE "Use Japanese commented headers" OFF)

# Select include directory based on language option
if(STATCPP_USE_JAPANESE)
    set(STATCPP_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include-ja)
    set(STATCPP_INSTALL_SOURCE_DIR include-ja)
    message(STATUS "statcpp: Using Japanese commented headers")
else()
    set(STATCPP_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set(STATCPP_INSTALL_SOURCE_DIR include)
    message(STATUS "statcpp: Using English headers")
endif()

# Define interface library as header-only library
add_library(statcpp INTERFACE)
target_include_directories(statcpp INTERFACE
    $<BUILD_INTERFACE:${STATCPP_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Enable compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(statcpp INTERFACE
        -Wall -Wextra -Wpedantic
    )
elseif(MSVC)
    target_compile_options(statcpp INTERFACE
        /W4
    )
endif()

# Build option for tests
option(STATCPP_BUILD_TESTS "Build tests" ON)

# Build option for sanitizers
option(STATCPP_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

if(STATCPP_ENABLE_SANITIZERS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(statcpp INTERFACE
            -fsanitize=address,undefined
            -fno-omit-frame-pointer
            -g
        )
        target_link_options(statcpp INTERFACE
            -fsanitize=address,undefined
        )
    endif()
endif()

if(STATCPP_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
    add_subdirectory(testWithR)
endif()

# Installation settings
include(GNUInstallDirs)
install(TARGETS statcpp
    EXPORT statcppTargets
)

# Install selected language version headers (preserving statcpp/ subdirectory)
install(DIRECTORY ${STATCPP_INSTALL_SOURCE_DIR}/statcpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT statcppTargets
    FILE statcppTargets.cmake
    NAMESPACE statcpp::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/statcpp
)

# Generate and install config file
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/statcppConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/statcppConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/statcpp
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/statcppConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/statcppConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/statcppConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/statcpp
)
