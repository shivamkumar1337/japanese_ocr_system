import nagisa
import pykakasi
from typing import Dict, List
from dataclasses import dataclass

try:
    from jamdict import Jamdict
    jam = Jamdict()
    HAS_JAMDICT = True
except:
    HAS_JAMDICT = False
    jam = None

# Initialize pykakasi
kks = pykakasi.kakasi()


@dataclass
class Token:
    """Represents a tokenized Japanese word"""
    text: str
    pos: str
    hiragana: str
    katakana: str
    romaji: str
    is_kanji: bool
    meaning: str = ""


class NLPAgent:
    """Agent responsible for Japanese NLP processing"""
    
    def __init__(self):
        self.name = "NLP Agent"
        self.description = "Tokenizes Japanese text and provides readings using nagisa and pykakasi"
        self.has_dictionary = HAS_JAMDICT
    
    def tokenize(self, text: str) -> List[Token]:
        """
        Tokenize Japanese text into words with readings
        
        Args:
            text: Japanese text to tokenize
            
        Returns:
            List of Token objects with readings and POS tags
        """
        print(f"[{self.name}] Tokenizing text ({len(text)} chars)")
        
        # Use nagisa for tokenization and POS tagging
        tokens_data = nagisa.tagging(text)
        result = []
        
        for word, pos in zip(tokens_data.words, tokens_data.postags):
            # Check if word contains kanji
            has_kanji = any('\u4e00' <= char <= '\u9fff' for char in word)
            
            # Get readings
            readings = self._convert_to_readings(word)
            
            # Get dictionary meaning if available
            meaning = self._get_dictionary_meaning(word) if has_kanji else ""
            
            result.append(Token(
                text=word,
                pos=pos,
                hiragana=readings["hiragana"],
                katakana=readings["katakana"],
                romaji=readings["romaji"],
                is_kanji=has_kanji,
                meaning=meaning
            ))
        
        kanji_count = sum(1 for t in result if t.is_kanji)
        print(f"[{self.name}] Created {len(result)} tokens ({kanji_count} kanji)")
        
        return result
    
    def _convert_to_readings(self, text: str) -> Dict[str, str]:
        """Convert Japanese text to hiragana, katakana, and romaji"""
        result = kks.convert(text)
        
        return {
            "hiragana": "".join([item['hira'] for item in result]),
            "katakana": "".join([item['kana'] for item in result]),
            "romaji": "".join([item['hepburn'] for item in result])
        }
    
    def _get_dictionary_meaning(self, word: str) -> str:
        """Get meaning from JMdict dictionary"""
        if not self.has_dictionary:
            return ""
        
        try:
            result = jam.lookup(word)
            for entry in result.entries:
                for sense in entry.senses:
                    glosses = [g.text for g in sense.gloss if g.text]
                    if glosses:
                        return "; ".join(glosses[:2])  # Top 2 meanings
            return ""
        except:
            return ""
