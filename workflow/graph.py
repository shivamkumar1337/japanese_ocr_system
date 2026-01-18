from langgraph.graph import StateGraph, END
from typing import Dict, Any
import uuid
import cv2
from datetime import datetime
from workflow.state import ProcessingState
from agents.visualization_agent import Annotation


class JapaneseTextWorkflow:
    """LangGraph workflow for Japanese text processing"""
    
    def __init__(self, ocr_agent, nlp_agent, llm_agent, viz_agent):
        self.ocr_agent = ocr_agent
        self.nlp_agent = nlp_agent
        self.llm_agent = llm_agent
        self.viz_agent = viz_agent
        
        # Build graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the processing workflow graph"""
        workflow = StateGraph(ProcessingState)
        
        # Add nodes
        workflow.add_node("ocr", self._ocr_node)
        workflow.add_node("nlp", self._nlp_node)
        workflow.add_node("llm", self._llm_node)
        workflow.add_node("visualize", self._visualize_node)
        
        # Define edges
        workflow.set_entry_point("ocr")
        workflow.add_edge("ocr", "nlp")
        workflow.add_edge("nlp", "llm")
        workflow.add_edge("llm", "visualize")
        workflow.add_edge("visualize", END)
        
        return workflow.compile()
    
    def _ocr_node(self, state: ProcessingState) -> ProcessingState:
        """OCR processing node"""
        print("\n[WORKFLOW] Node: OCR")
        
        result = self.ocr_agent.extract_text(state["image_path"])
        
        state["image_array"] = result.image
        state["ocr_elements"] = result.elements
        state["ocr_lines"] = result.lines
        state["full_text"] = result.full_text
        
        print(f"[WORKFLOW] OCR extracted {len(result.full_text)} characters, {len(result.elements)} elements, {len(result.lines)} lines")
        
        return state
    
    def _nlp_node(self, state: ProcessingState) -> ProcessingState:
        """NLP processing node"""
        print("\n[WORKFLOW] Node: NLP")
        
        tokens = self.nlp_agent.tokenize(state["full_text"])
        state["tokens"] = tokens
        
        # Build vocabulary dictionary from tokens
        vocabulary = {}
        for token in tokens:
            if token.is_kanji and token.meaning:
                vocabulary[token.text] = token.meaning
        
        state["vocabulary"] = vocabulary
        
        print(f"[WORKFLOW] NLP tokenized {len(tokens)} tokens, {len(vocabulary)} with meanings")
        
        # Show some example kanji tokens
        kanji_tokens = [t.text for t in tokens if t.is_kanji][:10]
        if kanji_tokens:
            print(f"[WORKFLOW] Example kanji: {kanji_tokens}")
        
        return state
    
    def _llm_node(self, state: ProcessingState) -> ProcessingState:
        """LLM analysis node"""
        print("\n[WORKFLOW] Node: LLM")
        
        analysis = self.llm_agent.analyze(state["full_text"])
        
        state["translation"] = analysis.translation
        state["grammar_patterns"] = analysis.grammar_patterns
        state["sentence_breakdown"] = []
        
        print(f"[WORKFLOW] LLM analysis: {len(analysis.grammar_patterns)} grammar patterns")
        
        return state
    
    def _visualize_node(self, state: ProcessingState) -> ProcessingState:
        """Visualization node - creates annotated image with furigana"""
        print("\n[WORKFLOW] Node: Visualization")
        
        # Validate image array
        if state["image_array"] is None:
            raise ValueError("Image array is None, cannot create visualizations")
        
        # Process OCR elements and find matching tokens
        annotations = []
        seen_positions = set()  # Track (x, y, text) to avoid duplicates
        
        print(f"[WORKFLOW] Processing {len(state['ocr_elements'])} OCR elements...")
        
        for elem in state["ocr_elements"]:
            elem_text = elem["text"]
            position_key = (elem["x"], elem["y"], elem_text)
            
            # Skip if we've already annotated this position with same text
            if position_key in seen_positions:
                continue
            
            # Check if element contains kanji
            has_kanji = any('\u4e00' <= char <= '\u9fff' for char in elem_text)
            
            if not has_kanji:
                continue
            
            print(f"[WORKFLOW] Processing OCR element: '{elem_text}' at ({elem['x']}, {elem['y']})")
            
            # Find matching token(s) that contain this kanji
            best_match = None
            best_match_length = 0
            
            for token in state["tokens"]:
                if token.is_kanji:
                    # Check if token matches or is contained in element
                    if token.text == elem_text:
                        best_match = token
                        best_match_length = len(token.text)
                        print(f"[WORKFLOW]   -> Exact match: '{token.text}' (hiragana: {token.hiragana})")
                        break
                    elif token.text in elem_text and len(token.text) > best_match_length:
                        best_match = token
                        best_match_length = len(token.text)
                        print(f"[WORKFLOW]   -> Partial match: '{token.text}' in '{elem_text}'")
                    elif elem_text in token.text and len(elem_text) > best_match_length:
                        # OCR element is part of a longer token
                        best_match = token
                        best_match_length = len(elem_text)
                        print(f"[WORKFLOW]   -> Element in token: '{elem_text}' in '{token.text}'")
            
            if best_match:
                # Get meaning
                meaning = state["vocabulary"].get(best_match.text, best_match.meaning)
                
                annotations.append(Annotation(
                    kanji=elem_text,
                    hiragana=best_match.hiragana,
                    meaning=meaning if meaning else "",
                    x=elem["x"],
                    y=elem["y"],
                    w=elem["w"],
                    h=elem["h"]
                ))
                seen_positions.add(position_key)
                print(f"[WORKFLOW]   ✓ Added annotation for '{elem_text}'")
            else:
                # If no match found, try to get reading directly from pykakasi
                print(f"[WORKFLOW]   ! No token match, getting direct reading...")
                try:
                    import pykakasi
                    kks = pykakasi.kakasi()
                    result = kks.convert(elem_text)
                    hiragana = "".join([item['hira'] for item in result])
                    
                    annotations.append(Annotation(
                        kanji=elem_text,
                        hiragana=hiragana,
                        meaning="",
                        x=elem["x"],
                        y=elem["y"],
                        w=elem["w"],
                        h=elem["h"]
                    ))
                    seen_positions.add(position_key)
                    print(f"[WORKFLOW]   ✓ Added direct annotation: '{elem_text}' -> '{hiragana}'")
                except Exception as e:
                    print(f"[WORKFLOW]   ✗ Failed to get reading: {e}")
        
        print(f"\n[WORKFLOW] Created {len(annotations)} total annotations")
        
        # Create annotated image with furigana
        try:
            annotated_img = self.viz_agent.annotate(state["image_array"], annotations)
            
            # Validate output
            if annotated_img is None:
                raise ValueError("Visualization agent returned None")
                
        except Exception as e:
            print(f"[WORKFLOW] Error in visualization: {e}")
            # Use original image as fallback
            annotated_img = state["image_array"].copy()
        
        # Save image with unique ID
        unique_id = uuid.uuid4().hex[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_filename = f"annotated_furigana_{timestamp}_{unique_id}.png"
        
        # Save with error handling
        try:
            success = cv2.imwrite(output_filename, annotated_img)
            
            if not success:
                print(f"[WORKFLOW] Warning: Failed to save annotated image")
                # Try different format
                output_filename = f"annotated_furigana_{timestamp}_{unique_id}.jpg"
                success = cv2.imwrite(output_filename, annotated_img)
                
            if not success:
                raise Exception("Failed to save image in any format")
            else:
                print(f"[WORKFLOW] Saved annotated image: {output_filename}")
                
        except Exception as e:
            print(f"[WORKFLOW] Error saving image: {e}")
            raise
        
        state["annotations"] = annotations
        state["annotated_image_readings_path"] = output_filename
        state["annotated_image_meanings_path"] = ""
        
        return state
    
    def process(self, image_path: str) -> Dict[str, Any]:
        """
        Process image through the workflow
        
        Args:
            image_path: Path to input image
            
        Returns:
            Processing results with annotated image
        """
        print(f"\n{'='*70}")
        print("🚀 STARTING JAPANESE TEXT PROCESSING WORKFLOW")
        print(f"{'='*70}")
        
        start_time = datetime.now()
        
        # Initialize state
        initial_state: ProcessingState = {
            "image_path": image_path,
            "image_array": None,
            "ocr_elements": [],
            "ocr_lines": [],
            "full_text": "",
            "tokens": [],
            "vocabulary": {},
            "translation": "",
            "grammar_patterns": [],
            "sentence_breakdown": [],
            "annotations": [],
            "annotated_image_readings_path": "",
            "annotated_image_meanings_path": "",
            "processing_time": "",
            "error": ""
        }
        
        try:
            # Run workflow
            final_state = self.graph.invoke(initial_state)
            
            # Calculate processing time
            end_time = datetime.now()
            processing_time = str(end_time - start_time)
            final_state["processing_time"] = processing_time
            
            # Format output
            result = self._format_output(final_state)
            
            print(f"\n{'='*70}")
            print("✅ WORKFLOW COMPLETE")
            print(f"⏱️  Time: {processing_time}")
            print(f"📝 Text: {len(final_state['full_text'])} chars")
            print(f"📚 Vocabulary: {len(final_state['vocabulary'])} words")
            print(f"🎨 Annotations: {len(final_state['annotations'])}")
            print(f"🖼️  Image: {final_state['annotated_image_readings_path']}")
            print(f"{'='*70}\n")
            
            return result
            
        except Exception as e:
            print(f"\n❌ WORKFLOW ERROR: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _format_output(self, state: ProcessingState) -> Dict[str, Any]:
        """Format final output"""
        # Convert annotations to vocabulary list
        vocab_list = []
        seen_kanji = set()
        
        for ann in state["annotations"]:
            if ann.kanji not in seen_kanji and ann.meaning:
                vocab_list.append({
                    "kanji": ann.kanji,
                    "hiragana": ann.hiragana,
                    "meaning": ann.meaning
                })
                seen_kanji.add(ann.kanji)
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "processing_time": state["processing_time"],
            "extracted_text": {
                "full_text": state["full_text"],
                "character_count": len(state["full_text"]),
                "elements_count": len(state["ocr_elements"]),
                "lines_count": len(state["ocr_lines"])
            },
            "vocabulary": vocab_list[:100],  # Limit to 100 words
            "analysis": {
                "translation": state["translation"],
                "grammar_patterns": state["grammar_patterns"],
                "sentence_breakdown": state["sentence_breakdown"]
            },
            "annotated_image": state["annotated_image_readings_path"],
            "stats": {
                "total_annotations": len(state["annotations"]),
                "vocabulary_words": len(vocab_list),
                "grammar_patterns": len(state["grammar_patterns"])
            }
        }