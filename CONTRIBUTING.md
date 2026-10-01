# Contributing to PlastiTrace

Thank you for contributing. PlastiTrace is a web-only stack: Next.js frontend + Flask API + PyTorch classifier.

## Prerequisites

| Tool | Version | Used for |
|------|---------|----------|
| Python | 3.11+ | API and ML inference |
| Node.js | 20+ | Web frontend (`web/`) |
| npm | 10+ | Web dependencies |

On macOS, install Xcode Command Line Tools if NumPy or OpenCV fail to build:

```bash
xcode-select --install
```

## Clone and setup

```bash
git clone https://github.com/will702/PlastiTrace.git
cd PlastiTrace

# Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Web frontend
cd web
cp .env.example .env.local
npm install
cd ..
```

Place model weights at `models/plastitrace.pth` if they are not already present.

## Run locally

**Terminal 1 - API:**

```bash
source venv/bin/activate
python api.py
```

**Terminal 2 - Frontend:**

```bash
cd web && npm run dev
```

Open `http://localhost:3000`. The frontend reads `NEXT_PUBLIC_API_URL` from `web/.env.local` (defaults to `http://localhost:5001`).

## Repo layout

| Path | Purpose |
|------|---------|
| `api.py` | Flask REST API (`/api/classify`, `/api/health`) |
| `ml/` | ResNet18 classifier, preprocessing, config |
| `models/` | Trained weights (`plastitrace.pth`) |
| `web/` | Next.js frontend (deploy to Vercel) |

See architecture diagrams in [README.md](README.md).

## Before submitting

1. **No secrets** - Do not commit `.env`, `.env.local`, API keys, or tokens.
2. **No cache files** - `__pycache__/`, `*.pyc`, `.DS_Store` must stay out of git.
3. **No agent tooling** - Do not commit `.cursor/`, `.agents/`, `.claude/`, `.gemini/`, or `skills-lock.json`.
4. **Web changes** - Run `cd web && npm run build` and fix any errors.
5. **Python check** - Verify: `python -c "from ml.classifier import PlastiTraceClassifier"`.
6. **Docs** - Update README.md if you change API routes, env vars, or project structure.

## Deployment notes

| Component | Host | Config |
|-----------|------|--------|
| Frontend | Vercel | Root directory: `web`, env: `NEXT_PUBLIC_API_URL` |
| API | Render / Railway / HF Spaces | Gunicorn; `debug=False`; restrict CORS |

Production API checklist:

- Restrict CORS to your Vercel domain
- Serve over HTTPS (required for mobile camera)
- Add upload size limits and file type validation

## Questions

Open a GitHub issue if setup steps fail or documentation is out of date.
