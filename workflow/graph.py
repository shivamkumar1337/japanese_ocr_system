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
        
        return state
    
    def _nlp_node(self, state: ProcessingState) -> ProcessingState:
        """NLP processing node"""
        print("\n[WORKFLOW] Node: NLP")
        
        tokens = self.nlp_agent.tokenize(state["full_text"])
        state["tokens"] = tokens
        
        return state
    
    def _llm_node(self, state: ProcessingState) -> ProcessingState:
        """LLM analysis node"""
        print("\n[WORKFLOW] Node: LLM")
        
        analysis = self.llm_agent.analyze(state["full_text"])
        
        # state["vocabulary"] = analysis.vocabulary
        state["translation"] = analysis.translation
        state["grammar_patterns"] = analysis.grammar_patterns
        # state["sentence_breakdown"] = analysis.sentence_breakdown
        
        return state
    
    def _visualize_node(self, state: ProcessingState) -> ProcessingState:
        """Visualization node"""
        print("\n[WORKFLOW] Node: Visualization")
        
        # Prepare annotations
        annotations = []
        for line in state["ocr_lines"]:
            line_text = "".join([elem["text"] for elem in line])
            line_tokens = self.nlp_agent.tokenize(line_text)
            
            for token in line_tokens:
                if token.is_kanji:
                    for elem in line:
                        if token.text in elem["text"]:
                            # Get meaning from vocabulary or token
                            meaning = state["vocabulary"].get(token.text, token.meaning)
                            if not meaning:
                                meaning = "meaning unavailable"
                            
                            annotations.append(Annotation(
                                kanji=token.text,
                                hiragana=token.hiragana,
                                meaning=meaning,
                                x=elem["x"],
                                y=elem["y"],
                                w=elem["w"],
                                h=elem["h"]
                            ))
                            break
        
        # Annotate image
        annotated_image = self.viz_agent.annotate(state["image_array"], annotations)
        
        # Save image
        output_filename = f"annotated_{uuid.uuid4().hex}.png"
        cv2.imwrite(output_filename, annotated_image)
        
        state["annotations"] = annotations
        state["annotated_image_path"] = output_filename
        
        return state
    
    def process(self, image_path: str) -> Dict[str, Any]:
        """
        Process image through the workflow
        
        Args:
            image_path: Path to input image
            
        Returns:
            Processing results
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
            "annotated_image_path": "",
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
            print(f"{'='*70}\n")
            
            return result
            
        except Exception as e:
            print(f"\n❌ WORKFLOW ERROR: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _format_output(self, state: ProcessingState) -> Dict[str, Any]:
        """Format final output"""
        # Convert annotations to dicts
        vocab_list = []
        for ann in state["annotations"][:50]:
            vocab_list.append({
                "kanji": ann.kanji,
                "hiragana": ann.hiragana,
                "meaning": ann.meaning
            })
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "processing_time": state["processing_time"],
            "extracted_text": {
                "full_text": state["full_text"],
                "character_count": len(state["full_text"]),
                "elements_count": len(state["ocr_elements"])
            },
            "vocabulary": vocab_list,
            "analysis": {
                "translation": state["translation"],
                "grammar_patterns": state["grammar_patterns"],
                "sentence_breakdown": state["sentence_breakdown"]
            },
            "annotated_image": state["annotated_image_path"]
        }
