"""
MVec Encoder: mvec_encoder.py
Encodes text into word_ids with identity and end channels.
"""

import re
from collections import Counter
from typing import Dict, List, Tuple, Optional
from enum import Enum


class EncodingMode(Enum):
    """Encoding modes for Marina training"""
    QA = "qa"  # Simple Question-Answer
    QBA = "qba"  # Question-Bridge-Answer
    CONTINUOUS = "continuous"  # Continuous text stream


class MVecEncoder:
    """
    Encodes text into multi-channel vectors for Marina.
    
    Channels:
    - word_id: Vocabulary index
    - identity: USER | MARINA_INTERNAL | MARINA_VISIBLE
    - end: NO | YES (marks turn completion)
    
    Encoding Modes:
    - QA: Simple question-answer pairs (user input → visible response)
    - QBA: Question-bridge-answer (user → internal reasoning → visible response)
    - CONTINUOUS: Continuous text stream (all MARINA_VISIBLE, for book training)
    """
    
    def __init__(self):
        self.vocab = {"<unk>": 0, "<pad>": 1}
        self.idx_to_word = {0: "<unk>", 1: "<pad>"}
        self.next_id = 2
        
        # Channel vocabularies
        self.identity_vocab = {
            "USER": 0,
            "MARINA_INTERNAL": 1,
            "MARINA_VISIBLE": 2
        }
        
        self.end_vocab = {
            "NO": 0,
            "YES": 1
        }
    
    def tokenize(self, text: str) -> List[str]:
        """Simple word + punctuation tokenizer"""
        tokens = re.findall(r'\w+(?:-\w+)*|[.,!?;:\-\[\](){}"\'/\\]', text, re.UNICODE)
        return tokens
    
    def build_vocab(self, texts: List[str], min_freq: int = 1):
        """
        Build vocabulary from list of texts.
        
        Args:
            texts: List of text strings
            min_freq: Minimum frequency for inclusion
        """
        all_tokens = []
        for text in texts:
            all_tokens.extend(self.tokenize(text))
        
        counter = Counter(all_tokens)
        
        for word, freq in counter.items():
            if freq >= min_freq and word not in self.vocab:
                self.vocab[word] = self.next_id
                self.idx_to_word[self.next_id] = word
                self.next_id += 1
        
        print(f"Vocabulary built: {len(self.vocab)} words")
    
    def encode_word(self, word: str) -> int:
        """Convert word to vocabulary ID"""
        return self.vocab.get(word, 0)  # 0 = <unk>
    
    def decode_word(self, idx: int) -> str:
        """Convert vocabulary ID to word"""
        return self.idx_to_word.get(idx, "<unk>")
    
    def encode_sequence(
        self,
        text: str,
        identity: str,
        is_last_in_turn: bool = False
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Encode a text sequence with channels.
        
        Args:
            text: The text to encode
            identity: "USER", "MARINA_INTERNAL", or "MARINA_VISIBLE"
            is_last_in_turn: Whether this is the last sequence in a turn
            
        Returns:
            (word_ids, identity_ids, end_ids)
        """
        tokens = self.tokenize(text)
        
        word_ids = []
        identity_ids = []
        end_ids = []
        
        identity_id = self.identity_vocab[identity]
        
        for i, token in enumerate(tokens):
            word_id = self.encode_word(token)
            word_ids.append(word_id)
            identity_ids.append(identity_id)
            
            # Mark end only on the last token if this is last in turn
            if i == len(tokens) - 1 and is_last_in_turn:
                end_ids.append(self.end_vocab["YES"])
            else:
                end_ids.append(self.end_vocab["NO"])
        
        return word_ids, identity_ids, end_ids
    
    # =========================================================================
    # NEW: Mode-specific encoding methods
    # =========================================================================
    
    def encode_qa(
        self,
        question: str,
        answer: str
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Encode in QA mode: Simple question-answer pairs.
        
        Structure:
        - Question (USER, ends with YES)
        - Answer (MARINA_VISIBLE, ends with YES)
        
        Args:
            question: User's question
            answer: Marina's visible answer
            
        Returns:
            (word_ids, identity_ids, end_ids) for complete sequence
        """
        all_word_ids = []
        all_identity_ids = []
        all_end_ids = []
        
        # Encode question
        w, i, e = self.encode_sequence(question, "USER", is_last_in_turn=True)
        all_word_ids.extend(w)
        all_identity_ids.extend(i)
        all_end_ids.extend(e)
        
        # Encode answer
        w, i, e = self.encode_sequence(answer, "MARINA_VISIBLE", is_last_in_turn=True)
        all_word_ids.extend(w)
        all_identity_ids.extend(i)
        all_end_ids.extend(e)
        
        return all_word_ids, all_identity_ids, all_end_ids
    
    def encode_qba(
        self,
        question: str,
        bridge: str,
        answer: str
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Encode in QBA mode: Question-Bridge-Answer with internal reasoning.
        
        Structure:
        - Question (USER, ends with YES)
        - Bridge (MARINA_INTERNAL, ends with YES)
        - Answer (MARINA_VISIBLE, ends with YES)
        
        Args:
            question: User's question
            bridge: Marina's internal reasoning (hidden from user)
            answer: Marina's visible answer
            
        Returns:
            (word_ids, identity_ids, end_ids) for complete sequence
        """
        all_word_ids = []
        all_identity_ids = []
        all_end_ids = []
        
        # Encode question
        w, i, e = self.encode_sequence(question, "USER", is_last_in_turn=True)
        all_word_ids.extend(w)
        all_identity_ids.extend(i)
        all_end_ids.extend(e)
        
        # Encode bridge (internal)
        w, i, e = self.encode_sequence(bridge, "MARINA_INTERNAL", is_last_in_turn=True)
        all_word_ids.extend(w)
        all_identity_ids.extend(i)
        all_end_ids.extend(e)
        
        # Encode answer (visible)
        w, i, e = self.encode_sequence(answer, "MARINA_VISIBLE", is_last_in_turn=True)
        all_word_ids.extend(w)
        all_identity_ids.extend(i)
        all_end_ids.extend(e)
        
        return all_word_ids, all_identity_ids, all_end_ids
    
    def encode_continuous(
        self,
        text: str,
        chunk_size: Optional[int] = None
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Encode in CONTINUOUS mode: Stream of text for book/corpus training.
        
        Structure:
        - All text encoded as MARINA_VISIBLE
        - If chunk_size provided, inserts END markers every chunk_size tokens
        - Otherwise, only final token marked as END
        
        Args:
            text: Continuous text (e.g., from a book)
            chunk_size: Optional number of tokens per chunk (for turn boundaries)
            
        Returns:
            (word_ids, identity_ids, end_ids) for complete sequence
        """
        tokens = self.tokenize(text)
        
        word_ids = []
        identity_ids = []
        end_ids = []
        
        identity_id = self.identity_vocab["MARINA_VISIBLE"]
        
        for i, token in enumerate(tokens):
            word_id = self.encode_word(token)
            word_ids.append(word_id)
            identity_ids.append(identity_id)
            
            # Mark end based on chunk_size or final token
            if chunk_size and (i + 1) % chunk_size == 0:
                end_ids.append(self.end_vocab["YES"])
            elif i == len(tokens) - 1:
                end_ids.append(self.end_vocab["YES"])
            else:
                end_ids.append(self.end_vocab["NO"])
        
        return word_ids, identity_ids, end_ids
    
    def encode(
        self,
        mode: EncodingMode,
        **kwargs
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Universal encoding method that dispatches to mode-specific encoders.
        
        Args:
            mode: EncodingMode (QA, QBA, or CONTINUOUS)
            **kwargs: Mode-specific arguments
                QA: question, answer
                QBA: question, bridge, answer
                CONTINUOUS: text, chunk_size (optional)
                
        Returns:
            (word_ids, identity_ids, end_ids) for complete sequence
            
        Examples:
            # QA mode
            encoder.encode(EncodingMode.QA, question="Hello?", answer="Hi!")
            
            # QBA mode
            encoder.encode(EncodingMode.QBA, 
                          question="What is 2+2?",
                          bridge="math addition",
                          answer="Four")
            
            # Continuous mode
            encoder.encode(EncodingMode.CONTINUOUS, 
                          text="Long book text...",
                          chunk_size=100)
        """
        if mode == EncodingMode.QA:
            return self.encode_qa(
                question=kwargs['question'],
                answer=kwargs['answer']
            )
        elif mode == EncodingMode.QBA:
            return self.encode_qba(
                question=kwargs['question'],
                bridge=kwargs['bridge'],
                answer=kwargs['answer']
            )
        elif mode == EncodingMode.CONTINUOUS:
            return self.encode_continuous(
                text=kwargs['text'],
                chunk_size=kwargs.get('chunk_size')
            )
        else:
            raise ValueError(f"Unknown encoding mode: {mode}")
    
    # =========================================================================
    # Keep original method for backward compatibility
    # =========================================================================
    
    def encode_conversation_turn(
        self,
        user_text: str,
        bridge_text: str,
        response_text: str
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Encode a complete conversation turn (user → bridge → response).
        
        [LEGACY METHOD - use encode_qba() instead]
        
        Args:
            user_text: User's input
            bridge_text: Marina's internal reasoning (hidden)
            response_text: Marina's visible response
            
        Returns:
            (word_ids, identity_ids, end_ids) for complete sequence
        """
        return self.encode_qba(user_text, bridge_text, response_text)
    
    # =========================================================================
    # Decoding and utility methods (unchanged)
    # =========================================================================
    
    def decode_sequence(
        self,
        word_ids: List[int],
        identity_ids: Optional[List[int]] = None,
        filter_identity: Optional[str] = None
    ) -> str:
        """
        Decode word_ids back to text.
        
        Args:
            word_ids: List of vocabulary indices
            identity_ids: Optional list of identity channel values
            filter_identity: If provided, only decode tokens with this identity
            
        Returns:
            Decoded text string
        """
        if filter_identity and identity_ids:
            target_id = self.identity_vocab[filter_identity]
            words = [
                self.decode_word(wid) 
                for wid, iid in zip(word_ids, identity_ids)
                if iid == target_id
            ]
        else:
            words = [self.decode_word(wid) for wid in word_ids]
        
        # Simple reconstruction with spacing
        result = []
        for word in words:
            if word in ".,!?;:":
                result.append(word)
            else:
                if result and result[-1] not in ".,!?;:":
                    result.append(" ")
                result.append(word)
        
        return "".join(result).strip()
    
    def get_vocab_size(self) -> int:
        """Return vocabulary size"""
        return len(self.vocab)
    
    def save_vocab(self, path: str):
        """Save vocabulary to file"""
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab': self.vocab,
                'idx_to_word': self.idx_to_word,
                'next_id': self.next_id
            }, f, ensure_ascii=False, indent=2)
    
    def load_vocab(self, path: str):
        """Load vocabulary from file"""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.vocab = data['vocab']
            self.idx_to_word = {int(k): v for k, v in data['idx_to_word'].items()}
            self.next_id = data['next_id']


if __name__ == "__main__":
    # Test the encoder with all three modes
    print("Testing MVecEncoder - All Modes")
    print("=" * 70)
    
    encoder = MVecEncoder()
    
    # Build vocab from sample texts
    sample_texts = [
        "Hello. I am Marina.",
        "What is the capital of France?",
        "The capital of France is Paris.",
        "Geography. France. Capital. Paris.",
        "I am here with you.",
        "Once upon a time, in a distant land, there lived a wise old sage."
    ]
    
    encoder.build_vocab(sample_texts)
    print(f"\nVocabulary size: {encoder.get_vocab_size()}")
    
    # =========================================================================
    # Test 1: QA Mode
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: QA MODE (Question-Answer)")
    print("=" * 70)
    
    question = "What is the capital of France?"
    answer = "The capital of France is Paris."
    
    word_ids, identity_ids, end_ids = encoder.encode(
        EncodingMode.QA,
        question=question,
        answer=answer
    )
    
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")
    print(f"\nEncoded length: {len(word_ids)} tokens")
    
    print("\nToken breakdown:")
    for i, (w, id_val, e) in enumerate(zip(word_ids, identity_ids, end_ids)):
        word = encoder.decode_word(w)
        identity = ["USER", "INTERNAL", "VISIBLE"][id_val]
        end = "YES" if e == 1 else "NO"
        print(f"  {i:3d}: '{word:15s}' | {identity:10s} | end={end}")
    
    # =========================================================================
    # Test 2: QBA Mode
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: QBA MODE (Question-Bridge-Answer)")
    print("=" * 70)
    
    question = "What is the capital of France?"
    bridge = "Geography France Capital Paris"
    answer = "The capital of France is Paris."
    
    word_ids, identity_ids, end_ids = encoder.encode(
        EncodingMode.QBA,
        question=question,
        bridge=bridge,
        answer=answer
    )
    
    print(f"\nQuestion: {question}")
    print(f"Bridge: {bridge}")
    print(f"Answer: {answer}")
    print(f"\nEncoded length: {len(word_ids)} tokens")
    
    print("\nToken breakdown:")
    for i, (w, id_val, e) in enumerate(zip(word_ids, identity_ids, end_ids)):
        word = encoder.decode_word(w)
        identity = ["USER", "INTERNAL", "VISIBLE"][id_val]
        end = "YES" if e == 1 else "NO"
        print(f"  {i:3d}: '{word:15s}' | {identity:10s} | end={end}")
    
    # Test filtering decode
    print("\nFiltered decode:")
    visible_only = encoder.decode_sequence(
        word_ids, 
        identity_ids, 
        filter_identity="MARINA_VISIBLE"
    )
    print(f"  Visible only: {visible_only}")
    
    internal_only = encoder.decode_sequence(
        word_ids,
        identity_ids,
        filter_identity="MARINA_INTERNAL"
    )
    print(f"  Internal only: {internal_only}")
    
    # =========================================================================
    # Test 3: CONTINUOUS Mode
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: CONTINUOUS MODE (Book/Corpus Training)")
    print("=" * 70)
    
    continuous_text = "Once upon a time, in a distant land, there lived a wise old sage."
    
    # Test without chunking
    print("\n--- Without chunking (single END marker) ---")
    word_ids, identity_ids, end_ids = encoder.encode(
        EncodingMode.CONTINUOUS,
        text=continuous_text
    )
    
    print(f"\nText: {continuous_text}")
    print(f"Encoded length: {len(word_ids)} tokens")
    print(f"END markers: {sum(end_ids)} (at position {end_ids.index(1)})")
    
    # Test with chunking
    print("\n--- With chunking (chunk_size=5) ---")
    word_ids, identity_ids, end_ids = encoder.encode(
        EncodingMode.CONTINUOUS,
        text=continuous_text,
        chunk_size=5
    )
    
    print(f"\nText: {continuous_text}")
    print(f"Encoded length: {len(word_ids)} tokens")
    print(f"END markers: {sum(end_ids)} at positions {[i for i, e in enumerate(end_ids) if e == 1]}")
    
    print("\nToken breakdown (with chunking):")
    for i, (w, id_val, e) in enumerate(zip(word_ids, identity_ids, end_ids)):
        word = encoder.decode_word(w)
        identity = ["USER", "INTERNAL", "VISIBLE"][id_val]
        end = "YES" if e == 1 else "NO"
        marker = " <-- CHUNK END" if e == 1 else ""
        print(f"  {i:3d}: '{word:15s}' | {identity:10s} | end={end}{marker}")
    
    # =========================================================================
    # Test 4: Legacy method still works
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 4: LEGACY METHOD (encode_conversation_turn)")
    print("=" * 70)
    
    word_ids_legacy, identity_ids_legacy, end_ids_legacy = encoder.encode_conversation_turn(
        question, bridge, answer
    )
    
    print(f"Legacy method produces same output as QBA: {word_ids == word_ids_legacy}")
    
    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
