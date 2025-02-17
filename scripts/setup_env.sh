#!/bin/bash

# 环境变量
PYTHON_VERSION="3.11"  # 使用系统安装的Python 3.11
VENV_NAME="solar-venv"
REQUIREMENTS_FILE="requirements.txt"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查Python版本
check_python() {
    echo -e "${YELLOW}Checking Python version...${NC}"
    if command -v python$PYTHON_VERSION &>/dev/null; then
        echo -e "${GREEN}Python $PYTHON_VERSION is installed${NC}"
        return 0
    else
        echo -e "${RED}Python $PYTHON_VERSION is not installed${NC}"
        return 1
    fi
}

# 创建虚拟环境
create_venv() {
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    cd "$PROJECT_ROOT"
    python$PYTHON_VERSION -m venv $VENV_NAME
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Virtual environment created successfully${NC}"
    else
        echo -e "${RED}Failed to create virtual environment${NC}"
        exit 1
    fi
}

# 激活虚拟环境
activate_venv() {
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    cd "$PROJECT_ROOT"
    source "$PROJECT_ROOT/$VENV_NAME/bin/activate"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Virtual environment activated${NC}"
        # 验证激活是否成功
        if python -c "import sys; sys.exit(0 if sys.prefix != sys.base_prefix else 1)"; then
            echo -e "${GREEN}Virtual environment verification successful${NC}"
        else
            echo -e "${RED}Virtual environment verification failed${NC}"
            exit 1
        fi
    else
        echo -e "${RED}Failed to activate virtual environment${NC}"
        exit 1
    fi
}

# 安装依赖
install_requirements() {
    echo -e "${YELLOW}Installing requirements...${NC}"
    # 升级pip
    pip install --no-cache-dir -U pip
    
    # 安装依赖
    pip install --no-cache-dir -r $REQUIREMENTS_FILE
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Requirements installed successfully${NC}"
    else
        echo -e "${RED}Failed to install requirements${NC}"
        exit 1
    fi
}

# 清理虚拟环境
clean_venv() {
    echo -e "${YELLOW}Cleaning up virtual environment...${NC}"
    cd "$PROJECT_ROOT"
    deactivate 2>/dev/null
    rm -rf $VENV_NAME
    echo -e "${GREEN}Cleanup completed${NC}"
}

# 显示内存使用情况
show_memory_usage() {
    echo -e "${YELLOW}Current memory usage:${NC}"
    free -h
}

# 显示Python和依赖信息
show_env_info() {
    echo -e "${YELLOW}Environment information:${NC}"
    which python
    python --version
    pip list
}

# 主函数
main() {
    case "$1" in
        "create")
            check_python
            create_venv
            activate_venv
            install_requirements
            show_memory_usage
            show_env_info
            ;;
        "activate")
            activate_venv
            show_env_info
            ;;
        "clean")
            clean_venv
            ;;
        *)
            echo "Usage: $0 {create|activate|clean}"
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@" 