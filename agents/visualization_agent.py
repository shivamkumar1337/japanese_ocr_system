from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from typing import List
from dataclasses import dataclass
import os


@dataclass
class Annotation:
    """Annotation to be drawn on image"""
    kanji: str
    hiragana: str
    meaning: str
    x: int
    y: int
    w: int
    h: int


class VisualizationAgent:
    """Agent responsible for image annotation"""
    
    def __init__(self):
        self.name = "Visualization Agent"
        self.description = "Annotates images with furigana and English meanings"
        self.font_paths = [
            "C:/Windows/Fonts/msgothic.ttc",
            "C:/Windows/Fonts/meiryo.ttc",
            "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
            "C:/Windows/Fonts/arial.ttf",
        ]
    
    def annotate(self, image: np.ndarray, annotations: List[Annotation]) -> np.ndarray:
        """
        Annotate image with furigana and meanings
        
        Args:
            image: Input image as numpy array
            annotations: List of Annotation objects
            
        Returns:
            Annotated image as numpy array
        """
        print(f"[{self.name}] Annotating {len(annotations)} words")
        
        # Convert to PIL for drawing
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image, 'RGBA')
        
        # Get fonts
        furigana_font = self._get_font(11)
        meaning_font = self._get_font(9)
        
        for ann in annotations:
            self._draw_annotation(draw, ann, furigana_font, meaning_font, pil_image.height)
        
        # Add border
        ImageDraw.Draw(pil_image).rectangle(
            [0, 0, pil_image.width-1, pil_image.height-1],
            outline=(100, 100, 100),
            width=2
        )
        
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def _draw_annotation(self, draw, ann: Annotation, furigana_font, meaning_font, img_height):
        """Draw single annotation (furigana + meaning)"""
        x, y, w, h = ann.x, ann.y, ann.w, ann.h
        
        try:
            # Draw furigana (above)
            furigana_bbox = draw.textbbox((0, 0), ann.hiragana, font=furigana_font)
            furigana_width = furigana_bbox[2] - furigana_bbox[0]
            furigana_height = furigana_bbox[3] - furigana_bbox[1]
            
            furigana_x = x + (w - furigana_width) // 2
            furigana_y = max(y - furigana_height - 5, 5)
            
            # White background
            draw.rectangle([
                furigana_x - 2, furigana_y - 2,
                furigana_x + furigana_width + 2, furigana_y + furigana_height + 2
            ], fill=(255, 255, 255, 220))
            
            # Red text
            draw.text((furigana_x, furigana_y), ann.hiragana, font=furigana_font, fill=(220, 0, 0, 255))
            
            # Draw meaning (below)
            meaning_text = ann.meaning[:30] + "..." if len(ann.meaning) > 30 else ann.meaning
            
            meaning_bbox = draw.textbbox((0, 0), meaning_text, font=meaning_font)
            meaning_width = meaning_bbox[2] - meaning_bbox[0]
            meaning_height = meaning_bbox[3] - meaning_bbox[1]
            
            meaning_x = x + (w - meaning_width) // 2
            meaning_y = y + h + 3
            
            # Adjust if near bottom
            if meaning_y + meaning_height + 5 > img_height:
                meaning_y = y - meaning_height - furigana_height - 8
            
            # White background
            draw.rectangle([
                meaning_x - 2, meaning_y - 2,
                meaning_x + meaning_width + 2, meaning_y + meaning_height + 2
            ], fill=(255, 255, 255, 200))
            
            # Blue text
            draw.text((meaning_x, meaning_y), meaning_text, font=meaning_font, fill=(0, 100, 200, 255))
            
        except Exception as e:
            print(f"[{self.name}] Error annotating '{ann.kanji}': {e}")
    
    def _get_font(self, size: int):
        """Get Japanese-compatible font"""
        for font_path in self.font_paths:
            if os.path.exists(font_path):
                try:
                    if font_path.endswith('.ttc'):
                        for index in [0, 1, 2]:
                            try:
                                return ImageFont.truetype(font_path, size, index=index)
                            except:
                                continue
                    else:
                        return ImageFont.truetype(font_path, size)
                except:
                    continue
        return ImageFont.load_default()
