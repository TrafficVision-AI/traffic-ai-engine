import os
import shutil
import uuid
import psycopg2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

# Hugging Face Compatibility: Force cache directories to /tmp to bypass "Read-only file system" errors
os.environ["YOLO_CONFIG_DIR"] = "/tmp"
os.environ["MPLCONFIGDIR"] = "/tmp"

# Environment Isolation: Dynamic referencing
DATABASE_URL = os.getenv("DATABASE_URL")
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://trafficvision-ai-dashboard.vercel.app")

app = FastAPI(title="Traffic Vision AI Engine", docs_url="/swagger")

# Global CORS Pipeline targeting our target deployments
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Initializing YOLO Model with Custom Roboflow Weights...")
model = YOLO("best.pt")

def save_to_neon(file_id: str, count: int):
    """Safely executes the metric mapping into Neon DB."""
    if not DATABASE_URL:
        print("DATABASE_URL not exported. Bypassing Neon DB synchronization.")
        return
        
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vehicle_analytics (
                id SERIAL PRIMARY KEY,
                video_url VARCHAR(255) NOT NULL,
                vehicle_count INT NOT NULL,
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        ''')
        
        video_path = f"/static/outputs/{file_id}.mp4"
        cursor.execute(
            "INSERT INTO vehicle_analytics (video_url, vehicle_count) VALUES (%s, %s)",
            (video_path, count)
        )
        
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Neon DB Interfacing Error: {e}")

@app.post("/api/v1/analyze")
async def analyze_video(file: UploadFile = File(...)):
    """Receives target MP4 buffer, executes ByteTrack tracking alongside YOLO detection, stores in DB."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Invalid Request: No file attached.")
        
    file_id = str(uuid.uuid4())
    ext = file.filename.split('.')[-1]
    input_path = f"/tmp/{file_id}_input.{ext}"
    
    # Store buffer into system /tmp memory securely
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Utilize tracking mechanisms to isolate individual cars correctly across distinct active frames via ByteTrack architecture
        results = model.track(source=input_path, tracker="bytetrack.yaml", stream=True, persist=True)
        
        vehicle_ids = set()
        vehicle_classes = [2, 3, 5, 7] # 2: Car, 3: Motorcycle, 5: Bus, 7: Truck
        
        for r in results:
            if r.boxes and r.boxes.id is not None:
                # Pair spatial boxing against the generated identity trackers across loops
                for box, track_id in zip(r.boxes, r.boxes.id):
                    if int(box.cls[0]) in vehicle_classes:
                        vehicle_ids.add(int(track_id))
        
        total_unique_vehicles = len(vehicle_ids)
        
        # Fire structural metadata out successfully into postgres
        save_to_neon(file_id, total_unique_vehicles)
        
        return {
            "success": True,
            "video_url": f"/static/outputs/{file_id}.mp4",
            "count": total_unique_vehicles,
            "message": "ByteTrack Analysis completed successfully."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean local environments
        if os.path.exists(input_path):
            os.remove(input_path)

@app.get("/docs")
async def health_check():
    """Manual fallback route designed exclusively to satisfy frontend ping mechanisms mapping ONLINE."""
    return {"status": "ONLINE", "message": "Traffic AI Engine operational."}

if __name__ == "__main__":
    import uvicorn
    # Hugging Face deployment pipeline forces internal PORT mappings typically binding to 7860
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
