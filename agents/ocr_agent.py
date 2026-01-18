import pytesseract
from pytesseract import Output
from PIL import Image
import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


@dataclass
class OCRResult:
    """OCR extraction result"""
    image: np.ndarray
    elements: List[Dict[str, Any]]
    lines: List[List[Dict[str, Any]]]
    full_text: str


class OCRAgent:
    """Agent responsible for optical character recognition"""
    
    def __init__(self):
        self.name = "OCR Agent"
        self.description = "Extracts Japanese text from images using Tesseract OCR"
    
    def extract_text(self, image_path: str) -> OCRResult:
        """
        Extract Japanese text from image with bounding boxes
        
        Args:
            image_path: Path to input image
            
        Returns:
            OCRResult containing extracted text and metadata
        """
        print(f"[{self.name}] Processing image: {image_path}")
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        img_np = np.array(image)
        
        # Perform OCR with Japanese language
        data = pytesseract.image_to_data(
            img_np,
            lang="jpn",
            output_type=Output.DICT,
            config='--psm 6 --oem 3'
        )
        
        # Extract text elements - lower threshold to catch more
        elements = []
        prev_text = None
        prev_y = None
        
        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            conf = float(data["conf"][i])
            
            # Lower threshold to catch all kanji
            if text and conf > 20:
                current_y = data["top"][i]
                
                # Skip if same text appears consecutively on the same line
                if prev_text == text and prev_y is not None and abs(current_y - prev_y) < 15:
                    continue
                
                elements.append({
                    "text": text,
                    "x": data["left"][i],
                    "y": data["top"][i],
                    "w": data["width"][i],
                    "h": data["height"][i],
                    "conf": conf
                })
                
                prev_text = text
                prev_y = current_y
        
        # Group elements into lines based on vertical position
        lines = self._group_into_lines(elements)
        
        # Combine all text maintaining original order
        full_text = " ".join([elem["text"] for elem in elements])
        
        print(f"[{self.name}] Extracted {len(elements)} text elements in {len(lines)} lines")
        
        return OCRResult(
            image=img_np,
            elements=elements,
            lines=lines,
            full_text=full_text
        )
    
    def _group_into_lines(self, elements: List[Dict]) -> List[List[Dict]]:
        """Group text elements into lines based on y-coordinate"""
        if not elements:
            return []
        
        lines = []
        current_line = []
        current_y = None
        
        for elem in sorted(elements, key=lambda e: (e["y"], e["x"])):
            if current_y is None:
                current_y = elem["y"]
                current_line.append(elem)
            else:
                # If within 15 pixels vertically, consider same line
                if abs(elem["y"] - current_y) < 15:
                    current_line.append(elem)
                else:
                    current_line.sort(key=lambda e: e["x"])
                    lines.append(current_line.copy())
                    current_line = [elem]
                    current_y = elem["y"]
        
        # Add the last line
        if current_line:
            current_line.sort(key=lambda e: e["x"])
            lines.append(current_line)
        
        return lines