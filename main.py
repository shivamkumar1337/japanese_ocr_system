from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile
import os
from datetime import datetime
import glob
import time

# Import agents
from agents.ocr_agent import OCRAgent
from agents.nlp_agent import NLPAgent
from agents.llm_agent import LLMAgent
from agents.visualization_agent import VisualizationAgent

# Import workflow
from workflow.graph import JapaneseTextWorkflow


# Initialize application
app = FastAPI(
    title="Japanese Text Processing API",
    description="Japanese OCR system with furigana annotations using OCR.space API",
)

# Initialize agents
print("🚀 Initializing agents...")
ocr_agent = OCRAgent()  # Uses OCR.space API
nlp_agent = NLPAgent()
llm_agent = LLMAgent()
viz_agent = VisualizationAgent()

print(f"✅ {ocr_agent.name} - {ocr_agent.description}")
print(f"✅ {nlp_agent.name} - {nlp_agent.description}")
print(f"✅ {llm_agent.name} - {llm_agent.description}")
print(f"✅ {viz_agent.name} - {viz_agent.description}")

# Initialize workflow
print("\n📊 Building LangGraph workflow...")
workflow = JapaneseTextWorkflow(ocr_agent, nlp_agent, llm_agent, viz_agent)
print("✅ Workflow ready")


@app.get("/")
async def root():
    """API information"""
    return {
        "name": "Japanese Text Processing API",
        "status": "operational",
        "version": "3.0",
        "architecture": "Multi-agent system with LangGraph orchestration",
        "ocr_engine": "OCR.space API",
        "agents": [
            {"name": ocr_agent.name, "description": ocr_agent.description},
            {"name": nlp_agent.name, "description": nlp_agent.description},
            {"name": llm_agent.name, "description": llm_agent.description},
            {"name": viz_agent.name, "description": viz_agent.description}
        ],
        "workflow": {
            "orchestrator": "LangGraph",
            "nodes": ["ocr", "nlp", "llm", "visualize"],
            "execution": "sequential"
        },
        "features": [
            "OCR text extraction (OCR.space API)",
            "Japanese tokenization (nagisa + pykakasi)",
            "Dictionary lookup (JMdict)",
            "LLM analysis (Groq)",
            "Furigana annotation between text lines"
        ],
        "output": {
            "annotated_image": "Original image with furigana readings positioned above kanji"
        },
        "endpoints": {
            "GET /": "API information",
            "POST /process": "Process Japanese text image with furigana annotations",
        }
    }


@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    """
    Process Japanese text image and return annotated image with furigana
    
    Returns JSON with:
    - extracted_text: Full text and statistics
    - vocabulary: List of kanji words with readings and meanings
    - analysis: Translation and grammar patterns from LLM
    - annotated_image: Path to annotated image with furigana
    - stats: Processing statistics
    """
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")):
        return JSONResponse(
            status_code=400,
            content={"error": "Only image files supported (PNG, JPG, JPEG, BMP, TIFF)"}
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        print(f"\n📥 Received file: {file.filename}")
        print(f"💾 Saved to: {tmp_path}")
        
        # Process through workflow
        result = workflow.process(tmp_path)
        
        # Cleanup temporary file
        os.unlink(tmp_path)
        print(f"🗑️  Cleaned up temporary file")
        
        return result
        
    except Exception as e:
        # Cleanup on error
        try:
            if 'tmp_path' in locals():
                os.unlink(tmp_path)
        except:
            pass
        
        print(f"❌ Error processing image: {e}")
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Processing failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        )


@app.on_event("startup")
async def startup_event():
    """Startup tasks"""
    print("\n" + "="*70)
    print("🚀 JAPANESE TEXT PROCESSING API")
    print("="*70)
    print(f"📖 Docs: http://localhost:8000/docs")
    print(f"🏗️  Architecture: Multi-agent with LangGraph")
    print(f"🔤 OCR Engine: OCR.space API")
    print(f"📚 Dictionary: {'✅ JMdict available' if nlp_agent.has_dictionary else '❌ JMdict not available'}")
    print("="*70)
    print("\n🔄 Workflow Pipeline:")
    print("   1. OCR Agent     → Extract text using OCR.space API")
    print("   2. NLP Agent     → Tokenize and get readings/meanings")
    print("   3. LLM Agent     → Translate and analyze grammar")
    print("   4. Viz Agent     → Add furigana between lines")
    print("="*70 + "\n")
    
    # Cleanup old annotated files (older than 1 hour)
    print("🧹 Cleaning up old files...")
    pattern = "annotated_furigana_*.png"
    cleaned = 0
    
    files = glob.glob(pattern)
    for file in files:
        try:
            if time.time() - os.path.getmtime(file) > 3600:
                os.remove(file)
                cleaned += 1
        except Exception as e:
            print(f"   Warning: Could not remove {file}: {e}")
    
    if cleaned > 0:
        print(f"   Removed {cleaned} old file(s)")
    else:
        print(f"   No old files to clean")
    


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)