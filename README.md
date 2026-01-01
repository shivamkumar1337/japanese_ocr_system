# 🇯🇵 Japanese Text Processing System - Production

A multi-agent system for processing Japanese text images with OCR, NLP analysis, dictionary lookup, LLM-powered grammar analysis, and visual annotation.

## ✨ Features

- 🔍 **OCR Text Extraction**: High-accuracy Japanese text extraction using Tesseract
- 📚 **Dictionary Integration**: JMdict-based vocabulary lookup with comprehensive meanings
- 🤖 **LLM Analysis**: Groq-powered grammar pattern explanations and natural translation
- 🎨 **Visual Annotation**: Furigana (hiragana readings) and English meanings overlaid on images
- 🔄 **LangGraph Orchestration**: Modern workflow management with state machines
- 🏗️ **Modular Architecture**: Independent, replaceable agent components

### Processing Pipeline

```
Image Upload → OCR Agent → NLP Agent → LLM Agent → Visualization Agent → Annotated Result
```

### Output Format

- **Extracted Text**: Full Japanese text with character positions
- **Vocabulary**: Kanji words with hiragana, katakana, romaji, and English meanings
- **Grammar Analysis**: Detailed explanations of particles, verb forms, and sentence patterns
- **Translation**: Natural English translation of the entire text
- **Annotated Image**: Original image with furigana above and meanings below kanji

## 🏗️ Architecture

### Multi-Agent System

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Application                   │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│           LangGraph Workflow Orchestrator                │
│  (State Machine with Sequential Node Execution)         │
└───────────────────────┬─────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┬─────────────┐
        │               │               │             │
        ▼               ▼               ▼             ▼
┌──────────────┐ ┌──────────────┐ ┌──────────┐ ┌──────────────┐
│  OCR Agent   │ │  NLP Agent   │ │ LLM Agent│ │ Viz Agent    │
│              │ │              │ │          │ │              │
│ Tesseract    │ │ nagisa       │ │ Groq API │ │ PIL/CV2      │
│ Text Extract │ │ pykakasi     │ │ Grammar  │ │ Furigana     │
│ Bounding Box │ │ JMdict       │ │ Translate│ │ Meanings     │
└──────────────┘ └──────────────┘ └──────────┘ └──────────────┘
```

### Agent Responsibilities

| Agent | Technology | Input | Output |
|-------|-----------|-------|--------|
| **OCR Agent** | Tesseract | Image file | Text elements with positions |
| **NLP Agent** | nagisa, pykakasi, JMdict | Raw text | Tokenized words with readings & meanings |
| **LLM Agent** | Groq API | Full text | Grammar patterns & translation |
| **Visualization Agent** | PIL, OpenCV | Image + annotations | Annotated image file |


### Required Software

#### 1. Tesseract OCR

**Windows:**
```bash
# Download installer from:
https://github.com/UB-Mannheim/tesseract/wiki
# During installation, make sure Japanese language data is selected
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-jpn
```

**Verify Installation:**
```bash
tesseract --version
tesseract --list-langs | grep jpn
```

## 🚀 Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/shivamkumar1337/japanese_ocr_system
cd japanese-ocr-system
```

### Step 2: Create Project Structure

```bash
# Create directory structure
mkdir -p agents workflow

# Create __init__.py files
touch agents/__init__.py
touch workflow/__init__.py
```

### Step 3: Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt


### Step 4: Download JMdict Dictionary Files

The system uses JMdict (Japanese-Multilingual Dictionary) for vocabulary lookup. You need to download the dictionary files before importing them.

#### Download Required Files

```bash
# Create dictionary directory
mkdir -p jmdict_data
cd jmdict_data

# Download JMdict (Japanese-English Dictionary)
curl -O https://www.edrdg.org/pub/Nihongo/JMdict_e.gz
gunzip JMdict_e.gz

# Download KanjiDic2 (Kanji Dictionary)
curl -O https://www.edrdg.org/pub/Nihongo/kanjidic2.xml.gz
gunzip kanjidic2.xml.gz

# Download JMnedict (Japanese Names Dictionary)
curl -O https://www.edrdg.org/pub/Nihongo/JMnedict.xml.gz
gunzip JMnedict.xml.gz

cd ..
```

**Alternative: Manual Download**

If curl doesn't work, download manually:

https://www.edrdg.org/pub/Nihongo/JMdict_e.gz
https://www.edrdg.org/pub/Nihongo/kanjidic2.xml.gz
https://www.edrdg.org/pub/Nihongo/JMnedict.xml.gz

1. Download the three files listed above
2. Place in `c:/users/user/.jamdict/data/` folder

#### Import Dictionary into jamdict

```bash
# Import the dictionary files
python -m jamdict import

# This will:
# - Create SQLite database from XML files
# - Index entries for fast lookup
# - Store in ~/.jamdict/ directory
# - Takes 2-5 minutes
```

### Step 5: Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# .env
# Groq API Configuration
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Tesseract Path (Windows only - adjust if needed)
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe

# Server Configuration
HOST=0.0.0.0
PORT=8000
```

**Get Groq API Key:**
1. Visit https://console.groq.com/
2. Sign up for free account
3. Generate API key from dashboard
4. Copy key to `.env` file


## ⚙️ Configuration

### Tesseract Configuration

If Tesseract is installed in a non-standard location, update in `agents/ocr_agent.py`:

```python
pytesseract.pytesseract.tesseract_cmd = r"YOUR_PATH_HERE"
```

### Font Configuration

For Japanese text rendering, the system looks for fonts in this order:

**Windows:**
- `C:/Windows/Fonts/msgothic.ttc`
- `C:/Windows/Fonts/meiryo.ttc`

**macOS:**
- `/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc`

**Linux:**
Install Japanese fonts:
```bash
sudo apt-get install fonts-noto-cjk fonts-noto-cjk-extra
```

## 🎯 Usage

### Starting the Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Server will start at: http://localhost:8000

### API Documentation

Once the server is running, visit:
- **Interactive Docs**: http://localhost:8000/docs


## 📚 API Documentation

### Endpoints

#### `GET /`
Get API information and system status.

**Response:**
```json
{
  "name": "Japanese Text Processing API - Production",
  "status": "operational",
  "architecture": "Multi-agent system with LangGraph orchestration"
}
```

#### `POST /process`
Process a Japanese text image.

**Parameters:**
- `file` (required): Image file (PNG, JPG, JPEG)