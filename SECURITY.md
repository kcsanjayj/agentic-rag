# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.x     | :white_check_mark: |
| 1.x     | :x:                |

## Security Features

Agentic-RAG implements the following security measures:

### API Key Handling
- **Session-only storage**: API keys are stored in memory only, never persisted to disk
- **No .env files**: Runtime configuration only - no environment files with secrets
- **Auto-clear on refresh**: API keys are cleared when user refreshes or starts new chat
- **Masked display**: API keys are masked (showing only first/last 4 chars) in UI

### Data Privacy
- **Local processing**: Document processing happens locally in your browser session
- **No persistent storage**: Chat history and documents are not stored on server
- **Secure transmission**: All API calls use HTTPS/TLS encryption

### Cloud Provider Security
- **User-provided keys only**: Users must provide their own API keys
- **No shared credentials**: Each user uses their own cloud provider accounts
- **No key logging**: API keys are never logged or exposed in error messages

## Reporting a Vulnerability

If you discover a security vulnerability, please report it by emailing:

**security@agentic-rag.dev** (replace with your actual email)

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We will respond within 48 hours and work to resolve the issue promptly.

## Security Best Practices for Users

1. **Never commit API keys** to git repositories
2. **Use environment-specific keys** for different deployments
3. **Rotate keys regularly** (every 90 days recommended)
4. **Monitor API usage** in your provider dashboards
5. **Enable 2FA** on your cloud provider accounts
6. **Use least-privilege keys** with minimal required permissions

## Known Limitations

- API keys are stored in browser memory (cleared on page refresh)
- Document content is processed by cloud LLM providers (users control which provider)
- No built-in encryption at rest (files stored temporarily during processing)

## Dependencies

We regularly update dependencies to patch security vulnerabilities:

```bash
pip install --upgrade -r requirements.txt
```

## Audit Log

| Date | Issue | Status |
|------|-------|--------|
| 2025-04-16 | Removed .env file dependency | :white_check_mark: Fixed |
| 2025-04-16 | Implemented session-only API keys | :white_check_mark: Fixed |

---

**Last Updated**: April 16, 2026
