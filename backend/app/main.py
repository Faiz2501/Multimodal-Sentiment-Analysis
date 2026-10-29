from pathlib import Path

from fastapi import FastAPI,File,Form,HTTPException,UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from app.core.audio import MAX_UPLOAD_BYTES,as_wav
from app.core.services import predictor,status,transcriber

app=FastAPI(title="Multimodal Sentiment Analysis API",version="1.0.0",description="Local speech and transcript inference API")
app.add_middleware(CORSMiddleware,allow_origins=["http://localhost:5173","http://127.0.0.1:5173"],allow_credentials=True,allow_methods=["GET","POST"],allow_headers=["*"])

@app.get("/api/health")
def health(): return status()

@app.post("/api/transcribe")
async def transcribe(audio:UploadFile=File(...)):
    payload=await audio.read(MAX_UPLOAD_BYTES+1)
    try:
        with as_wav(payload,audio.filename or "recording.wav") as wav:
            result=transcriber().transcribe(str(wav))
        return {"text":result}
    except ValueError as exc: raise HTTPException(status_code=422,detail=str(exc)) from exc
    except Exception as exc: raise HTTPException(status_code=503,detail=f"Local transcription could not start: {exc}") from exc

@app.post("/api/analyze")
async def analyze(audio:UploadFile=File(...),text:str=Form(...)):
    if not text.strip(): raise HTTPException(status_code=422,detail="Transcript text is required")
    try: service=predictor()
    except FileNotFoundError as exc: raise HTTPException(status_code=503,detail="The trained model bundle is not installed. Add the checkpoint, config, and tokenizer under artifacts/final; see the README.") from exc
    payload=await audio.read(MAX_UPLOAD_BYTES+1)
    try:
        with as_wav(payload,audio.filename or "recording.wav") as wav:
            result=service.predict(str(wav),text.strip())
        return result
    except ValueError as exc: raise HTTPException(status_code=422,detail=str(exc)) from exc
    except Exception as exc: raise HTTPException(status_code=500,detail=f"Analysis failed: {exc}") from exc

@app.exception_handler(Exception)
async def unexpected_error(request,exc):
    return JSONResponse(status_code=500,content={"detail":"Unexpected server error"})


# In production, serve the built React app from this API origin. During development, Vite serves the UI and proxies /api requests.
FRONTEND_DIST=Path(__file__).resolve().parents[2]/"frontend"/"dist"
if (FRONTEND_DIST/"index.html").is_file():
    app.mount("/",StaticFiles(directory=FRONTEND_DIST,html=True),name="frontend")
