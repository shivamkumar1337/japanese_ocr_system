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
        self.description = "Annotates images with furigana readings"
        self.font_paths = [
            "C:/Windows/Fonts/msgothic.ttc",
            "C:/Windows/Fonts/meiryo.ttc",
            "C:/Windows/Fonts/YuGothic.ttc",
            "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "C:/Windows/Fonts/arial.ttf",
        ]
    
    def annotate(self, image: np.ndarray, annotations: List[Annotation]) -> np.ndarray:
        """
        Create annotated image with furigana readings between lines
        
        Args:
            image: Input image as numpy array
            annotations: List of Annotation objects
            
        Returns:
            Annotated image as numpy array
        """
        print(f"[{self.name}] Creating annotated image for {len(annotations)} words")
        
        # Validate input image
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Invalid input image: must be a numpy array")
        
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"Invalid image shape: {image.shape}, expected (H, W, 3)")
        
        try:
            # Convert BGR to RGB for PIL
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image, 'RGBA')
            
            # Calculate line spacing to determine font size
            line_spacing = self._calculate_line_spacing(annotations)
            
            # Get appropriately sized font (smaller if lines are close)
            if line_spacing < 30:
                font_size = 8
            elif line_spacing < 50:
                font_size = 10
            else:
                font_size = 12
            
            furigana_font = self._get_font(font_size)
                        
            # Draw annotations
            for ann in annotations:
                try:
                    self._draw_furigana(draw, ann, furigana_font, pil_image.height)
                except Exception as e:
                    print(f"[{self.name}] Warning: Failed to draw furigana for '{ann.kanji}': {e}")
                    continue
            
            # Add title at top
            title_font = self._get_font(12)
            self._add_title(draw, pil_image.width, "Furigana Readings", title_font)
            
            # Convert back to BGR for OpenCV
            result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            print(f"[{self.name}] ✅ Created annotated image: shape {result.shape}")
            
            return result
            
        except Exception as e:
            print(f"[{self.name}] Error creating annotated image: {e}")
            # Return original image on error
            return image.copy()
    
    def _calculate_line_spacing(self, annotations: List[Annotation]) -> int:
        """Calculate average spacing between text lines"""
        if len(annotations) < 2:
            return 50  # Default spacing
        
        # Get unique y positions (lines)
        y_positions = sorted(set(ann.y for ann in annotations))
        
        if len(y_positions) < 2:
            return 50
        
        # Calculate average spacing
        spacings = []
        for i in range(len(y_positions) - 1):
            spacings.append(y_positions[i + 1] - y_positions[i])
        
        avg_spacing = sum(spacings) / len(spacings) if spacings else 50
        return int(avg_spacing)
    
    def _draw_furigana(self, draw, ann: Annotation, font, img_height):
        """Draw furigana reading above kanji, positioned in the space between lines"""
        x, y, w, h = ann.x, ann.y, ann.w, ann.h
        
        # Validate coordinates
        if w <= 0 or h <= 0:
            return
        
        # Get furigana dimensions
        furigana_bbox = draw.textbbox((0, 0), ann.hiragana, font=font)
        furigana_width = furigana_bbox[2] - furigana_bbox[0]
        furigana_height = furigana_bbox[3] - furigana_bbox[1]
        
        # Center horizontally above the kanji
        furigana_x = x + (w - furigana_width) // 2
        
        # Position in the space above the text line
        vertical_padding = 3
        furigana_y = y - furigana_height - vertical_padding
        
        # Ensure it doesn't go above the image
        if furigana_y < 20:
            furigana_y = 20
        
        # Semi-transparent white background
        padding = 2
        draw.rectangle([
            furigana_x - padding, furigana_y - padding,
            furigana_x + furigana_width + padding, furigana_y + furigana_height + padding
        ], fill=(255, 255, 255, 200))
        
        # Red text for readings
        draw.text((furigana_x, furigana_y), ann.hiragana, font=font, fill=(220, 0, 0, 255))
    
    def _add_title(self, draw, width, title_text, font):
        """Add title at the top of the image"""
        try:
            title_bbox = draw.textbbox((0, 0), title_text, font=font)
            title_width = title_bbox[2] - title_bbox[0]
            title_height = title_bbox[3] - title_bbox[1]
            
            title_x = (width - title_width) // 2
            title_y = 3
            
            # Background
            padding = 4
            draw.rectangle([
                title_x - padding, title_y - padding,
                title_x + title_width + padding, title_y + title_height + padding
            ], fill=(50, 50, 50, 230))
            
            # White text
            draw.text((title_x, title_y), title_text, font=font, fill=(255, 255, 255, 255))
        except Exception as e:
            print(f"[{self.name}] Warning: Could not add title: {e}")
    
    def _get_font(self, size: int):
        """Get Japanese-compatible font"""
        for font_path in self.font_paths:
            if os.path.exists(font_path):
                try:
                    if font_path.endswith('.ttc'):
                        for index in [0, 1, 2, 3]:
                            try:
                                return ImageFont.truetype(font_path, size, index=index)
                            except:
                                continue
                    else:
                        return ImageFont.truetype(font_path, size)
                except Exception as e:
                    continue
        
        # Fallback to default
        print(f"[{self.name}] Warning: Using default font, Japanese may not render correctly")
        return ImageFont.load_default()