#!/bin/bash
# Script to generate Doxygen documentation for statcpp

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}statcpp Documentation Generator${NC}"
echo "=================================="
echo ""

# Check if Doxygen is installed
if ! command -v doxygen &> /dev/null; then
    echo -e "${RED}Error: Doxygen is not installed${NC}"
    echo "Please install Doxygen:"
    echo "  macOS: brew install doxygen"
    echo "  Ubuntu: sudo apt-get install doxygen"
    echo "  Windows: Download from https://www.doxygen.nl/download.html"
    exit 1
fi

echo -e "${GREEN}✓${NC} Doxygen found: $(doxygen --version)"
echo ""

# Check if Doxyfile exists
if [ ! -f "Doxyfile" ]; then
    echo -e "${RED}Error: Doxyfile not found${NC}"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Clean previous documentation
if [ -d "doxydocs" ]; then
    echo -e "${YELLOW}Cleaning previous documentation...${NC}"
    rm -rf doxydocs
fi

# Generate documentation
echo -e "${GREEN}Generating documentation...${NC}"
doxygen Doxyfile

# Check if generation was successful
if [ -d "doxydocs/html" ]; then
    echo ""
    echo -e "${GREEN}✓ Documentation generated successfully!${NC}"
    echo ""
    echo "HTML documentation: doxydocs/html/index.html"
    echo ""
    echo "To view the documentation:"
    echo "  open doxydocs/html/index.html"
    echo ""
else
    echo -e "${RED}Error: Documentation generation failed${NC}"
    exit 1
fi
