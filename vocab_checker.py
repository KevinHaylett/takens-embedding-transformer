"""
vocab_checker.py
Check if text/questions contain out-of-vocabulary (OOV) words
Can be used standalone or imported into other modules
"""

import json
import re
from typing import List, Tuple, Dict, Optional
from collections import Counter


class VocabChecker:
    """
    Check text against a Marina vocabulary file.
    Helps identify OOV words before inference.
    """
    
    def __init__(self, vocab_path: Optional[str] = None):
        """
        Initialize the vocab checker.
        
        Args:
            vocab_path: Path to vocab JSON file (can be loaded later)
        """
        self.vocab = {}
        self.vocab_size = 0
        self.vocab_path = None
        
        if vocab_path:
            self.load_vocab(vocab_path)
    
    def load_vocab(self, vocab_path: str):
        """
        Load vocabulary from JSON file.
        
        Args:
            vocab_path: Path to vocab JSON file
        """
        with open(vocab_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.vocab = data['vocab']
            self.vocab_size = len(self.vocab)
            self.vocab_path = vocab_path
        
        print(f"✓ Loaded vocabulary: {self.vocab_size:,} words")
        print(f"  From: {vocab_path}")
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using Marina's tokenization pattern.
        Must match mvec_encoder.py tokenization exactly!
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # This MUST match mvec_encoder.py line 53
        tokens = re.findall(r'\w+(?:-\w+)*|[.,!?;:\-\[\](){}"\'/\\]', text, re.UNICODE)
        return tokens
    
    def check_text(
        self, 
        text: str, 
        verbose: bool = True
    ) -> Tuple[List[str], List[str], Dict]:
        """
        Check text for OOV words.
        
        Args:
            text: Text to check
            verbose: Print detailed report
            
        Returns:
            (in_vocab_tokens, oov_tokens, stats_dict)
        """
        if not self.vocab:
            raise ValueError("No vocabulary loaded! Call load_vocab() first.")
        
        tokens = self.tokenize(text)
        
        in_vocab = []
        oov = []
        
        for token in tokens:
            if token in self.vocab:
                in_vocab.append(token)
            else:
                oov.append(token)
        
        # Statistics
        total = len(tokens)
        oov_count = len(oov)
        coverage = (total - oov_count) / total * 100 if total > 0 else 0.0
        
        stats = {
            'total_tokens': total,
            'in_vocab_count': len(in_vocab),
            'oov_count': oov_count,
            'coverage_pct': coverage,
            'unique_oov': len(set(oov))
        }
        
        if verbose:
            self._print_report(text, tokens, in_vocab, oov, stats)
        
        return in_vocab, oov, stats
    
    def check_questions(
        self, 
        questions: List[str], 
        verbose: bool = True
    ) -> Dict:
        """
        Check a list of questions for OOV words.
        
        Args:
            questions: List of question strings
            verbose: Print detailed report
            
        Returns:
            Dictionary with results for each question
        """
        if not self.vocab:
            raise ValueError("No vocabulary loaded! Call load_vocab() first.")
        
        results = {}
        all_oov = []
        
        for i, question in enumerate(questions):
            in_vocab, oov, stats = self.check_text(question, verbose=False)
            results[i] = {
                'question': question,
                'in_vocab': in_vocab,
                'oov': oov,
                'stats': stats
            }
            all_oov.extend(oov)
        
        if verbose:
            self._print_questions_report(results, all_oov)
        
        return results
    
    def is_safe_for_inference(self, text: str, allow_oov: bool = False) -> bool:
        """
        Check if text is safe for inference (no OOV words).
        
        Args:
            text: Text to check
            allow_oov: If True, return True even with OOV words
            
        Returns:
            True if safe for inference, False otherwise
        """
        _, oov, _ = self.check_text(text, verbose=False)
        
        if not allow_oov and oov:
            return False
        return True
    
    def suggest_alternatives(
        self, 
        oov_word: str, 
        max_suggestions: int = 5
    ) -> List[Tuple[str, int]]:
        """
        Suggest similar words from vocabulary for an OOV word.
        Uses simple Levenshtein-like scoring.
        
        Args:
            oov_word: Out-of-vocabulary word
            max_suggestions: Maximum number of suggestions
            
        Returns:
            List of (word, score) tuples
        """
        if not self.vocab:
            raise ValueError("No vocabulary loaded!")
        
        # Simple scoring: exact case match > case-insensitive > starts with > contains
        candidates = []
        oov_lower = oov_word.lower()
        
        for word in self.vocab.keys():
            if word in ['<unk>', '<pad>']:
                continue
            
            word_lower = word.lower()
            
            # Scoring
            if word == oov_word:
                score = 100
            elif word_lower == oov_lower:
                score = 90
            elif word.startswith(oov_word) or oov_word.startswith(word):
                score = 70
            elif word_lower.startswith(oov_lower) or oov_lower.startswith(word_lower):
                score = 60
            elif oov_lower in word_lower or word_lower in oov_lower:
                score = 40
            else:
                continue
            
            candidates.append((word, score))
        
        # Sort by score, then alphabetically
        candidates.sort(key=lambda x: (-x[1], x[0]))
        
        return candidates[:max_suggestions]
    
    def _print_report(
        self, 
        text: str, 
        tokens: List[str], 
        in_vocab: List[str], 
        oov: List[str], 
        stats: Dict
    ):
        """Print detailed check report"""
        print("=" * 70)
        print("VOCABULARY CHECK")
        print("=" * 70)
        print(f"\nText: {text}")
        print(f"\nTokens: {tokens}")
        print(f"\nStatistics:")
        print(f"  Total tokens: {stats['total_tokens']}")
        print(f"  In vocabulary: {stats['in_vocab_count']}")
        print(f"  Out of vocabulary: {stats['oov_count']}")
        print(f"  Coverage: {stats['coverage_pct']:.1f}%")
        
        if oov:
            print(f"\n⚠ WARNING: {len(set(oov))} unique OOV words found!")
            print("\nOOV Words:")
            for word in set(oov):
                count = oov.count(word)
                print(f"  • '{word}' (appears {count}x)")
                
                # Show suggestions
                suggestions = self.suggest_alternatives(word, max_suggestions=3)
                if suggestions:
                    print(f"    Suggestions: {', '.join(f'{w}' for w, _ in suggestions)}")
        else:
            print("\n✓ All tokens in vocabulary - safe for inference!")
        
        print("=" * 70)
    
    def _print_questions_report(self, results: Dict, all_oov: List[str]):
        """Print report for multiple questions"""
        print("=" * 70)
        print("VOCABULARY CHECK - MULTIPLE QUESTIONS")
        print("=" * 70)
        
        for i, data in results.items():
            q = data['question']
            oov = data['oov']
            stats = data['stats']
            
            status = "✓" if not oov else "⚠"
            print(f"\n{status} Question {i+1}: {q}")
            print(f"   Coverage: {stats['coverage_pct']:.1f}% | "
                  f"In vocab: {stats['in_vocab_count']} | OOV: {stats['oov_count']}")
            
            if oov:
                print(f"   OOV words: {', '.join(set(oov))}")
        
        # Summary
        total_questions = len(results)
        questions_with_oov = sum(1 for r in results.values() if r['oov'])
        unique_oov = len(set(all_oov))
        
        print("\n" + "─" * 70)
        print("SUMMARY")
        print("─" * 70)
        print(f"Total questions: {total_questions}")
        print(f"Questions with OOV: {questions_with_oov}")
        print(f"Unique OOV words: {unique_oov}")
        
        if unique_oov > 0:
            print(f"\n⚠ Found {unique_oov} unique OOV words:")
            oov_freq = Counter(all_oov)
            for word, count in oov_freq.most_common():
                print(f"  • '{word}' ({count}x)")
                suggestions = self.suggest_alternatives(word, max_suggestions=3)
                if suggestions:
                    print(f"    → {', '.join(f'{w}' for w, _ in suggestions)}")
        else:
            print("\n✓ All questions are safe for inference!")
        
        print("=" * 70)
    
    def get_vocab_stats(self) -> Dict:
        """Get vocabulary statistics"""
        if not self.vocab:
            return {}
        
        return {
            'vocab_size': self.vocab_size,
            'vocab_path': self.vocab_path,
            'special_tokens': [k for k in self.vocab.keys() if k.startswith('<')]
        }


# ============================================================
# STANDALONE MODE - Edit test cases here
# ============================================================

if __name__ == "__main__":
    
    # ========================================
    # CONFIGURATION - Edit these!
    # ========================================
    
    VOCAB_PATH = r'c:\Marina\models\corpus_ancora\corpus_ancoraC_vocab.json'
    
    # Test questions - PASTE YOUR QUESTIONS HERE
    TEST_QUESTIONS = [
        "Where is Lady Serendipity?",
        "How do I find Kaevin the Listener?",
        "Where are the mice in the walls?",
        "What is the meaning of Simul Pariter?",
        "Tell me about the Attralucians.",
    ]
    
    # Single text test - PASTE TEXT HERE
    TEST_TEXT = "Where is Lady Serendipity wandering today?"
    
    # ========================================
    # RUN CHECKS
    # ========================================
    
    print("\n" + "=" * 70)
    print("MARINA VOCABULARY CHECKER")
    print("=" * 70)
    
    # Initialize checker
    checker = VocabChecker()
    
    try:
        checker.load_vocab(VOCAB_PATH)
    except FileNotFoundError:
        print(f"\n✗ ERROR: Could not find vocab file at:")
        print(f"  {VOCAB_PATH}")
        print("\nPlease update VOCAB_PATH in this script.")
        exit(1)
    
    # Print vocab stats
    print("\nVocabulary Statistics:")
    stats = checker.get_vocab_stats()
    print(f"  Size: {stats['vocab_size']:,} words")
    print(f"  Special tokens: {', '.join(stats['special_tokens'])}")
    
    # Check single text
    print("\n" + "=" * 70)
    print("TEST 1: Single Text Check")
    print("=" * 70)
    checker.check_text(TEST_TEXT, verbose=True)
    
    # Check multiple questions
    print("\n" + "=" * 70)
    print("TEST 2: Multiple Questions Check")
    print("=" * 70)
    results = checker.check_questions(TEST_QUESTIONS, verbose=True)
    
    # Test the safety check function
    print("\n" + "=" * 70)
    print("TEST 3: Inference Safety Check")
    print("=" * 70)
    
    for question in TEST_QUESTIONS[:3]:
        is_safe = checker.is_safe_for_inference(question)
        status = "✓ SAFE" if is_safe else "✗ UNSAFE"
        print(f"{status}: {question}")
    
    print("\n" + "=" * 70)
    print("Tests complete!")
    print("=" * 70)
    
    # Show how to use in other modules
    print("\n" + "=" * 70)
    print("USAGE IN OTHER MODULES")
    print("=" * 70)
    print("""
# Import the checker
from vocab_checker import VocabChecker

# Initialize and load vocab
checker = VocabChecker('path/to/vocab.json')

# Quick safety check before inference
if checker.is_safe_for_inference(user_question):
    # Proceed with inference
    answer = generate_answer(user_question)
else:
    # Handle OOV words
    _, oov, _ = checker.check_text(user_question, verbose=False)
    print(f"Warning: Question contains OOV words: {oov}")
    # Decide how to handle (reject, warn user, etc.)
    """)
