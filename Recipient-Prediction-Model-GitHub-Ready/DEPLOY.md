# Deploy the Seva Recipient Forecast web app

## Recommended architecture

- **GitHub** stores this repository.
- **Railway (or another Python-capable host)** runs `webapp.py`.
- The **custom domain** points to the deployed Python service, not GitHub Pages.

GitHub Pages is static hosting and cannot run the scikit-learn model.

## Railway deployment

1. Push this repository to GitHub.
2. In Railway, create a new project from the GitHub repository.
3. Railway can build from the included `Dockerfile`.
4. Deploy the service.
5. Open the generated Railway URL and verify `/health` returns `{"status":"ok"}`.
6. In Railway, add your custom domain.
7. Update the domain's DNS records exactly as Railway instructs.
8. Remove the custom domain from GitHub Pages if it is currently attached there.

## Privacy

Do not commit a real Seva Guest Visit export. The dashboard is designed for a user to upload the current export at forecast time. The application code processes that upload in memory and does not write it to the repository.

If this site will be publicly reachable, add authentication before using real recipient data. At minimum, keep the deployment private or password-protected until authentication is added.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
flask --app webapp run --debug
```

Open `http://127.0.0.1:5000`.
