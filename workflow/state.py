from typing import TypedDict, List, Dict, Any, Optional
import numpy as np


class ProcessingState(TypedDict, total=False):
    """State object passed through the workflow"""
    # Input
    image_path: str
    
    # OCR outputs
    image_array: Optional[np.ndarray]
    ocr_elements: List[Dict[str, Any]]
    ocr_lines: List[List[Dict[str, Any]]]
    full_text: str
    
    # NLP outputs
    tokens: List[Any]
    
    # LLM outputs
    vocabulary: Dict[str, str]
    translation: str
    grammar_patterns: List[str]
    sentence_breakdown: List[str]
    
    # Visualization outputs
    annotations: List[Any]
    annotated_image_readings_path: str
    annotated_image_meanings_path: str
    
    # Metadata
    processing_time: str
    error: str