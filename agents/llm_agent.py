import requests
import re
from typing import Dict, List, Any
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


@dataclass
class LLMAnalysis:
    """LLM analysis result"""
    translation: str
    grammar_patterns: List[str]


class LLMAgent:
    """Agent responsible for LLM-based comprehensive analysis"""
    
    def __init__(self):
        self.name = "LLM Agent"
        self.description = "Provides grammar analysis and translation using Groq LLM"
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY environment variable is not set. Set it in the environment or .env file.")
        self.headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
    
    def analyze(self, full_text: str) -> LLMAnalysis:
        """
        Comprehensive analysis of Japanese text
        Args:
            full_text: Complete Japanese text
        Returns:
            LLMAnalysis with translation and grammar patterns
        """
        print(f"[{self.name}] Analyzing text with LLM")
        
        prompt = f"""Analyze this Japanese text for language learners.

        TEXT:
        {full_text}

        TASK:
        Provide analysis in this format:

        TRANSLATION:
        [Natural English translation - translate the entire text naturally]

        GRAMMAR_PATTERNS:
        - Pattern (e.g., について): Explanation of usage and meaning
        - Pattern: Detailed explanation
        [List all important grammar patterns found in the text with detailed explanations]

        Be thorough and educational. Focus on explaining grammar patterns clearly."""

        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert Japanese teacher. Provide translation and grammar explanations for language learners."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 5000
        }
        
        try:
            response = requests.post(
                GROQ_API_URL, 
                headers=self.headers, 
                json=payload, 
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            llm_response = result["choices"][0]["message"]["content"]
            
            parsed = self._parse_response(llm_response)
            
            print(f"[{self.name}] Analysis complete: {len(parsed.grammar_patterns)} grammar patterns")
            
            return parsed
            
        except Exception as e:
            print(f"[{self.name}] Error: {e}")
            return LLMAnalysis(
                translation="Analysis unavailable",
                grammar_patterns=[]
            )
    
    def _parse_response(self, response_text: str) -> LLMAnalysis:
        result = {
            "translation": "",
            "grammar_patterns": []
        }
        
        lines = response_text.split('\n')
        current_section = None
        
        translation_lines = []
        grammar_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            line_upper = line.upper()
            
            if "TRANSLATION" in line_upper:
                current_section = "translation"
                continue
            elif "GRAMMAR" in line_upper:
                current_section = "grammar"
                continue
            
            # Parse content based on current section
            if current_section == "translation":
                # Skip section headers and bullet points for translation
                if not line.startswith("-") and ":" not in line and "[" not in line and "]" not in line:
                    translation_lines.append(line)
                    
            elif current_section == "grammar" and line.startswith("-"):
                grammar_lines.append(line[1:].strip())
        
        # Join translation lines
        result["translation"] = " ".join(translation_lines)
        
        # Clean up grammar patterns
        cleaned_grammar = []
        for pattern in grammar_lines:
            # Remove any markdown formatting
            pattern = pattern.replace("**", "").replace("*", "").strip()
            if pattern and not pattern.startswith("["):
                cleaned_grammar.append(pattern)
        
        result["grammar_patterns"] = cleaned_grammar
        
        # Fallback if parsing fails
        if not result["translation"] or not result["grammar_patterns"]:
            return self._fallback_parsing(response_text)
        
        return LLMAnalysis(**result)
    
    def _fallback_parsing(self, response_text: str) -> LLMAnalysis:
        """Fallback parsing if structured parsing fails"""
        # Try to extract translation - look for English text after "Translation" marker
        translation = ""
        grammar_patterns = []
        
        # Simple regex for translation
        trans_match = re.search(r'TRANSLATION[:\s]*(.*?)(?=GRAMMAR_PATTERNS|GRAMMAR|$)', 
                               response_text, re.IGNORECASE | re.DOTALL)
        if trans_match:
            translation = trans_match.group(1).strip()
        
        # Simple regex for grammar patterns
        grammar_match = re.search(r'GRAMMAR[:\s]*(.*?)(?=$)', 
                                 response_text, re.IGNORECASE | re.DOTALL)
        if grammar_match:
            grammar_text = grammar_match.group(1)
            # Extract lines starting with dash or bullet
            patterns = re.findall(r'[-•*]\s*(.*?)(?=\n[-•*]|\n\n|$)', grammar_text, re.DOTALL)
            grammar_patterns = [p.strip() for p in patterns if p.strip()]
        
        # If still no translation, use first paragraph as fallback
        if not translation:
            paragraphs = [p.strip() for p in response_text.split('\n\n') if p.strip()]
            if paragraphs:
                translation = paragraphs
        
        return LLMAnalysis(
            translation=translation,
            grammar_patterns=grammar_patterns
        )