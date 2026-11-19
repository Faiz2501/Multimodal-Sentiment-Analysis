# Deployment

## Recommended split

- **Frontend:** Vercel, with the Vercel project root set to `frontend`.
- **Backend:** Google Cloud Run from this repository root, using the included `Dockerfile`.

Render's free web service is not a suitable host for this model: its free service has 512 MB RAM and 0.1 CPU, while inference needs both pretrained encoders and the Python runtime in memory. Cloud Run is request-based and has a monthly free allowance, but it requires a billing-enabled Google Cloud project and usage beyond the free allowance can be charged. Set a maximum of one instance and zero minimum instances for a small demonstration; this limits scale but does not create a guaranteed billing cap.

## Backend on Cloud Run

1. Push this repository to GitHub and sign in to Google Cloud Console.
2. Create or select a Google Cloud project, enable billing, and enable Cloud Run and Cloud Build.
3. Deploy from the repository root with the included Dockerfile. Use 1 vCPU, 4 GiB memory, concurrency 1, minimum instances 0, maximum instances 1, and a request timeout suitable for cold model loading (up to 900 seconds).
4. Set `CORS_ORIGINS` to the exact Vercel website origin, such as `https://your-project.vercel.app`.
5. The container listens on Cloud Run's `PORT`. It caches Hugging Face model files under `/tmp/huggingface`; that cache is temporary and may be downloaded again after a cold start or new instance.
6. Confirm `/api/health` reports `model_ready: true`, then test transcription and a complete audio-plus-text prediction.

The checkpoint bundle is included in the repository. The matching pretrained DistilBERT, Wav2Vec2, and Whisper weights are downloaded from Hugging Face when first used by a new container instance.

## Frontend on Vercel

1. Import the GitHub repository in Vercel and set **Root Directory** to `frontend`.
2. Use `npm run build` as the build command and `dist` as the output directory. Install with `npm ci` when `package-lock.json` is present.
3. Add the build-time environment variable `VITE_API_BASE_URL` with the Cloud Run URL followed by `/api`, for example `https://your-api.run.app/api`.
4. Deploy, copy the Vercel production origin, and ensure that exact origin is included in the backend's `CORS_ORIGINS` value.
5. Redeploy the frontend after changing `VITE_API_BASE_URL`; it is embedded at build time.

For local development, leave `VITE_API_BASE_URL` unset: the frontend continues to call `/api` through the Vite development proxy.

## Use the Vercel frontend with a backend on your computer

This option serves the UI from Vercel but performs inference on your computer. It only works in a browser running on the same computer as the backend; other visitors' browsers would try to contact their own computer.

1. In the Vercel project settings, set the Production environment variable `VITE_API_BASE_URL` to `http://127.0.0.1:8001/api`, then redeploy the frontend.
2. Start the backend on your computer using the command below from the repository root in **Command Prompt**:

   ```cmd
   set PYTHONPATH=%CD%\src
   set CORS_ORIGINS=https://multimodal-sentiment-analysis-meld.vercel.app
   .venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8001
   ```

3. Open `https://multimodal-sentiment-analysis-meld.vercel.app`. If the browser asks whether the site can access devices on your local network, allow it. Keep the backend command window open while using the site.

Alternatively, leave the Vercel environment variable unset and run both frontend and backend locally; Vite proxies `/api` to port 8001.
