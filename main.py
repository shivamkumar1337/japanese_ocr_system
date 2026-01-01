from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
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
    title="Japanese Text Processing API - Production",
    description="Production-ready Japanese OCR system with LangGraph workflow",
)

# Initialize agents
print("🚀 Initializing agents...")
ocr_agent = OCRAgent()
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
        "name": "Japanese Text Processing API - Production",
        "status": "operational",
        "architecture": "Multi-agent system with LangGraph orchestration",
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
            "OCR text extraction (Tesseract)",
            "Japanese tokenization (nagisa + pykakasi)",
            "Dictionary lookup (JMdict)",
            "LLM analysis (Groq)",
            "Image annotation (PIL)"
        ],
        "endpoints": {
            "POST /process": "Process Japanese text image",
        }
    }


@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    """
    Returns:
    - Extracted text
    - Vocabulary with meanings
    - Grammar and translation analysis
    - Annotated image
    """
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        return JSONResponse(
            status_code=400,
            content={"error": "Only PNG, JPG, JPEG images supported"}
        )
    
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Process through workflow
        result = workflow.process(tmp_path)
        
        # Cleanup
        os.unlink(tmp_path)
        
        return result
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Processing failed: {str(e)}"}
        )


@app.on_event("startup")
async def startup_event():
    """Startup tasks"""
    print("\n" + "="*70)
    print("🚀 JAPANESE TEXT PROCESSING API - PRODUCTION")
    print("="*70)
    print(f"🌐 Server: http://localhost:8000")
    print(f"📖 Docs: http://localhost:8000/docs")
    print(f"🏗️  Architecture: Multi-agent with LangGraph")
    print(f"📚 Dictionary: {'✅' if nlp_agent.has_dictionary else '❌'}")
    print("="*70)
    print("\n🔄 Workflow Pipeline:")
    print("   OCR Agent → NLP Agent → LLM Agent → Visualization Agent")
    print("="*70 + "\n")
    
    # Cleanup old files
    files = glob.glob("annotated_*.png")
    for file in files:
        try:
            if time.time() - os.path.getmtime(file) > 3600:
                os.remove(file)
        except:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)