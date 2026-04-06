#!/bin/bash

# Agentic-RAG Setup Script
# This script sets up the development environment for Agentic-RAG

set -e  # Exit on any error

echo "🚀 Setting up Agentic-RAG development environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
check_python() {
    print_status "Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_success "Python $PYTHON_VERSION found"
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_VERSION=$(python --version | cut -d' ' -f2)
        print_success "Python $PYTHON_VERSION found"
        PYTHON_CMD="python"
    else
        print_error "Python is not installed. Please install Python 3.11 or higher."
        exit 1
    fi
    
    # Check Python version
    if [[ $PYTHON_VERSION < 3.11 ]]; then
        print_warning "Python version $PYTHON_VERSION is below recommended 3.11"
    fi
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    
    if [ ! -d "venv" ]; then
        $PYTHON_CMD -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    print_success "Virtual environment activated"
}

# Install Python dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Dependencies installed"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    directories=(
        "data/uploads"
        "data/chroma_db"
        "data/cache"
        "logs"
        "backups"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_status "Created directory: $dir"
        fi
    done
    
    print_success "Directories created"
}

# Setup environment file
setup_env() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_success "Environment file created from example"
            print_warning "Please edit .env file with your configuration"
        else
            print_error ".env.example not found"
            exit 1
        fi
    else
        print_warning ".env file already exists"
    fi
}

# Download embedding model (optional)
download_models() {
    print_status "Downloading embedding models..."
    
    # This would download the sentence transformer model
    # For now, we'll just create a placeholder
    print_status "Models will be downloaded on first use"
}

# Run initial tests
run_tests() {
    print_status "Running initial tests..."
    
    if command -v pytest &> /dev/null; then
        pytest tests/ -v --tb=short || print_warning "Some tests failed"
    else
        print_warning "pytest not found, skipping tests"
    fi
}

# Setup Git hooks (optional)
setup_git_hooks() {
    print_status "Setting up Git hooks..."
    
    if [ -d ".git" ]; then
        # Create pre-commit hook
        cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook for Agentic-RAG

echo "Running pre-commit checks..."

# Run linting
if command -v flake8 &> /dev/null; then
    flake8 backend/ --max-line-length=100 --ignore=E203,W503
fi

# Run formatting check
if command -v black &> /dev/null; then
    black --check backend/ --line-length=100
fi

echo "Pre-commit checks completed"
EOF
        
        chmod +x .git/hooks/pre-commit
        print_success "Git hooks setup completed"
    else
        print_warning "Not a Git repository, skipping hooks"
    fi
}

# Create startup script
create_startup_script() {
    print_status "Creating startup script..."
    
    cat > start.sh << 'EOF'
#!/bin/bash

# Agentic-RAG Startup Script

echo "Starting Agentic-RAG..."

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export PYTHONPATH=$(pwd)

# Start the application
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
EOF
    
    chmod +x start.sh
    print_success "Startup script created"
}

# Print next steps
print_next_steps() {
    echo ""
    print_success "🎉 Setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Edit the .env file with your configuration"
    echo "2. Add your OpenAI API key if you want to use LLM features"
    echo "3. Run the application with: ./start.sh"
    echo "4. Open http://localhost:8000 in your browser"
    echo ""
    echo "Useful commands:"
    echo "- Run tests: make test"
    echo "- Format code: make format"
    echo "- Lint code: make lint"
    echo "- Clear data: make clear-db"
    echo ""
    echo "For more information, see README.md"
}

# Main setup function
main() {
    echo "🔧 Agentic-RAG Development Environment Setup"
    echo "=========================================="
    echo ""
    
    check_python
    create_venv
    install_dependencies
    create_directories
    setup_env
    download_models
    run_tests
    setup_git_hooks
    create_startup_script
    print_next_steps
    
    echo ""
    print_success "Setup completed! 🚀"
}

# Run the setup
main "$@"
