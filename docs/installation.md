# Installation Guide

Get the Agentic RAG system up and running in minutes with this comprehensive installation guide.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.9 or higher
- **Node.js**: 18.0 or higher  
- **PostgreSQL**: 13.0 or higher (optional, for production)
- **Redis**: 6.0 or higher (optional, for caching)

### Development Tools
- **Git**: For version control
- **Docker**: For containerized deployment (optional)
- **Make**: For unified command interface

## 🚀 Quick Installation

### Method 1: Using Make (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/agentic-rag.git
cd agentic-rag

# Install dependencies
make dev-install

# Start development server
make serve
```

### Method 2: Manual Installation

```bash
# Clone the repository
git clone https://github.com/your-org/agentic-rag.git
cd agentic-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -e .[dev]

# Install frontend dependencies
cd frontend
npm install
cd ..

# Set up environment
cp .env.example .env
# Edit .env with your settings

# Start the server
python start.py
```

## ⚙️ Environment Configuration

### Required Environment Variables

Create a `.env` file in the project root:

```bash
# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=true

# AI Provider Configuration
AI_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here

# Database Configuration (Production)
DATABASE_URL=postgresql://user:password@localhost:5432/agentic_rag

# Cache Configuration (Optional)
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=your_secret_key_here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Logging
LOG_LEVEL=INFO
```

### Gemini API Setup

1. Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Add it to your `.env` file:
   ```bash
   GEMINI_API_KEY=AIzaSy...
   ```

## 🗄️ Database Setup

### Development (SQLite)
For development, the system uses SQLite by default - no setup required.

### Production (PostgreSQL)

#### Option 1: Docker
```bash
# Start PostgreSQL
docker run --name postgres-db \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=agentic_rag \
  -p 5432:5432 \
  -d postgres:15

# Run migrations
make migrate
```

#### Option 2: Local Installation
```bash
# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib  # Ubuntu/Debian
brew install postgresql  # macOS

# Create database
sudo -u postgres createdb agentic_rag

# Run migrations
make migrate
```

## 🌐 Frontend Setup

### Development Mode
```bash
cd frontend
npm run dev
```

The frontend will be available at `http://localhost:3000`

### Production Build
```bash
cd frontend
npm run build
```

The built files will be in `frontend/dist/`

## 🧪 Testing Setup

### Run All Tests
```bash
make test
```

### Run Specific Test Types
```bash
# Unit tests only
make test-unit

# Integration tests only
make test-integration

# Frontend tests
cd frontend && npm test
```

### Test Coverage
```bash
make coverage
```

Coverage report will be generated in `htmlcov/`

## 🐳 Docker Installation

### Using Docker Compose
```bash
# Clone and navigate to project
git clone https://github.com/your-org/agentic-rag.git
cd agentic-rag

# Start all services
docker-compose up -d

# Run migrations
docker-compose exec backend alembic upgrade head

# View logs
docker-compose logs -f
```

### Build Custom Image
```bash
# Build the image
docker build -t agentic-rag .

# Run the container
docker run -p 8000:8000 agentic-rag
```

## 🔧 Development Setup

### Pre-commit Hooks
```bash
# Install pre-commit hooks
make dev-install  # This installs hooks automatically

# Or manually
pre-commit install
```

### Code Quality Tools
```bash
# Format code
make format

# Run linting
make lint

# Type checking
make type-check
```

### IDE Configuration

#### VSCode
Install these extensions:
- Python
- TypeScript and JavaScript Language Features
- ESLint
- Prettier
- GitLens

#### PyCharm
1. Open the project directory
2. Configure Python interpreter to use the virtual environment
3. Enable pytest as the test runner

## 🚀 Verification

### Health Check
```bash
curl http://localhost:8000/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

### Configuration Check
```bash
curl http://localhost:8000/api/v1/config
```

### Test Query
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this system about?", "max_context": 5, "similarity_threshold": 0.1}'
```

## 🐛 Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>
```

#### Permission Denied
```bash
# Make scripts executable
chmod +x start.py
chmod +x run.bat
```

#### Module Not Found
```bash
# Reinstall dependencies
pip install -e .[dev]
```

#### Frontend Build Fails
```bash
# Clear npm cache
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

### Debug Mode
Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python start.py
```

### Database Issues
```bash
# Reset database
docker-compose down -v
docker-compose up -d
make migrate
```

## 📚 Next Steps

After successful installation:

1. [Read the API Documentation](api.md)
2. [Try Example Queries](user-guide/query-examples.md)
3. [Learn About the Quality System](quality.md)
4. [Deploy to Production](deployment.md)

## 🆘 Getting Help

- **Documentation**: [Full Docs](https://agentic-rag.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/your-org/agentic-rag/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/agentic-rag/discussions)

---

*Having trouble? Check the [troubleshooting guide](operations/troubleshooting.md) or open an issue.*
