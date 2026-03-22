# Deploying QuBrain to Render

Render is an excellent platform for hosting this full-stack application. Because the architecture consists of a Python/FastAPI backend and a React/Vite frontend, you will deploy **two separate services** on Render.

Both services can be run completely free using Render's Free Tier.

---

## 1. Deploy the Backend (Web Service)

The backend handles the machine learning model, PennyLane circuits, and SHAP explainability.

1. Go to your Render Dashboard and click **New +** > **Web Service**.
2. Connect your GitHub repository.
3. Set the following configuration for the Web Service:
   - **Name:** `qubrain-backend` (or any name you prefer)
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** `Free`
4. Expand **Advanced** and add the required **Environment Variables**:
   - `ADMIN_USERNAME`: `your-secure-username`
   - `ADMIN_PASSWORD_HASH`: `your-generate-sha256-hash` (You must use SHA-256 for the password hash here!)
   - `SECRET_KEY`: `a-long-random-string-like-6a8b1c...`
   - `ALGORITHM`: `HS256`
   - `QBRAIN_CORS_ORIGINS`: **Paste your Frontend URL here** *(e.g., `https://qubrain-frontend.onrender.com`)*
   - `PYTHON_VERSION`: `3.11` (highly recommended to ensure compatibility with Torch and PennyLane)
5. Click **Create Web Service**.

> **Note on Free Tier:** The free tier sleeps after 15 minutes of inactivity. When a user first opens the app after it sleeps, the backend may take up to 50 seconds to spin back up and load the PyTorch models into memory.

---

## 2. Deploy the Frontend (Static Site)

The frontend is a static React application built with Vite. It needs to know the URL of your newly deployed backend.

1. First, wait for your backend to finish deploying and **copy its URL** (e.g., `https://qubrain-backend-xyz.onrender.com`).
2. Go back to the Render dashboard and click **New +** > **Static Site**.
3. Connect the same GitHub repository.
4. Set the following configuration for the Static Site:
   - **Name:** `qubrain-frontend`
   - **Root Directory:** `frontend` *(This is critical!)*
   - **Build Command:** `npm ci && npm run build`
   - **Publish Directory:** `frontend/dist`
5. Expand **Advanced** and add the frontend **Environment Variable**:
   - `VITE_API_URL`: **Paste your backend URL here** *(Do not include a trailing slash, e.g. `https://qubrain-backend-xyz.onrender.com`)*
6. Click **Create Static Site**.

---

## 3. Verify Deployment

Once both services are live:
1. Open your Static Site URL (e.g., `https://qubrain-frontend.onrender.com`).
2. The UI should load immediately.
3. Click the **"Assess Patient"** or **"Random Patient"** button.
4. If it successfully loads the data and generates a prediction with SHAP explanations, the frontend and backend are communicating perfectly!

### Troubleshooting

- **CORS Errors:** If you see "Blocked by CORS" in your browser console, ensure that the `VITE_API_URL` exactly matches the URL of your backend. FastAPI is configured dynamically to allow the origin that the request comes from, so CORS should resolve automatically as long as the URL is correct.
- **Memory Errors During Build:** The backend builds `torch` and `pennylane`, which are heavy. Render's free tier occasionally hits memory limits during `pip install`. If the build fails, simply click **Manual Deploy > Clear build cache & deploy** and try again.
