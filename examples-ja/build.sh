#!/bin/bash
# Build script for statcpp examples (Japanese version)

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}statcpp Examples Build Script${NC}"
echo "=================================="
echo ""

# Create build directory
BUILD_DIR="build"
if [ -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}Cleaning previous build directory...${NC}"
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo -e "${GREEN}Configuring with CMake...${NC}"
cmake ..

# Build all examples
echo -e "${GREEN}Building examples...${NC}"
make -j$(sysctl -n hw.ncpu)

# Check if build was successful
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ All examples built successfully!${NC}"
    echo ""
    echo "Executables are in: $(pwd)"
    echo ""
    echo "To run an example:"
    echo "  cd build"
    echo "  ./example_basic_statistics"
    echo ""
else
    echo -e "${RED}Error: Build failed${NC}"
    exit 1
fi
