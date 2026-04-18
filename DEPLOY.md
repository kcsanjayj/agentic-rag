# PRODUCTION DEPLOYMENT GUIDE

## Quick Deploy Options

### Option 1: Railway (RECOMMENDED - Easiest)

1. Push code to GitHub
2. Go to [railway.app](https://railway.app)
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repo
5. Add environment variables:
   ```
   API_SECRET=your-secure-random-key
   ENV=production
   ```
6. Deploy!

Railway will automatically use `railway.json` configuration.

---

### Option 2: Render

1. Push code to GitHub
2. Go to [render.com](https://render.com)
3. Click "New Web Service"
4. Connect your GitHub repo
5. Use `render.yaml` configuration
6. Deploy!

---

### Option 3: Docker (Local / VPS)

```bash
# Build
docker build -t agentic-rag .

# Run
docker run -p 8000:8000 \
  -e API_SECRET=your-secret-key \
  -e ENV=production \
  agentic-rag
```

---

### Option 4: Docker Compose (Local Development)

```bash
# Create .env file
echo "API_SECRET=your-secret-key" > .env

# Run
docker-compose up
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_SECRET` | ✅ | Internal API authentication |
| `ENV` | ❌ | `production` or `development` |
| `LOG_LEVEL` | ❌ | `INFO`, `DEBUG`, `WARNING` |
| `HOST` | ❌ | Default: `0.0.0.0` |
| `PORT` | ❌ | Default: `8000` |

---

## API Usage

### Headers Required

```http
X-Api-Key: your-internal-api-secret
X-User-Api-Key: sk-openai-key-here
X-Session-Id: user-session-123  # Optional, for memory
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/query` | POST | Standard query |
| `/api/v1/query-stream` | POST | Streaming (SSE) |
| `/api/v1/query-pro` | POST | Agentic with tools |
| `/api/v1/upload` | POST | Document upload |

---

## Production Checklist

- [ ] Set strong `API_SECRET`
- [ ] Enable rate limiting (already configured)
- [ ] Set up monitoring (Railway/Render provide this)
- [ ] Configure CORS if needed (in `main.py`)
- [ ] Set memory limits (in `railway.json`/`render.yaml`)
- [ ] Test with real OpenAI API key

---

## Troubleshooting

### Port already in use
Change port mapping: `-p 8080:8000`

### Memory issues
Reduce workers in Dockerfile: `--workers 1`

### Slow responses
- Enable streaming endpoint `/query-stream`
- Use smaller chunk sizes

### API key errors
Ensure `X-User-Api-Key` starts with `sk-`
