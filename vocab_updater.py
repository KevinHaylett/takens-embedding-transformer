"""
vocab_updater.py
Utility to add new words to an existing vocabulary file.

WARNING: Only use this for adding small numbers of words!
For large changes, rebuild the vocab from scratch.
"""

import json
from typing import List


class VocabUpdater:
    """
    Update an existing vocabulary file by adding new words.
    """
    
    def __init__(self, vocab_path: str):
        """
        Load an existing vocabulary file.
        
        Args:
            vocab_path: Path to vocab JSON file
        """
        self.vocab_path = vocab_path
        
        with open(vocab_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.vocab = data['vocab']
            self.idx_to_word = {int(k): v for k, v in data['idx_to_word'].items()}
            self.next_id = data['next_id']
        
        print(f"✓ Loaded vocabulary: {len(self.vocab):,} words")
        print(f"  From: {vocab_path}")
        print(f"  Next available ID: {self.next_id}")
    
    def add_words(self, words: List[str]) -> int:
        """
        Add new words to vocabulary.
        
        Args:
            words: List of words to add
            
        Returns:
            Number of words actually added (excludes duplicates)
        """
        added_count = 0
        
        for word in words:
            if word in self.vocab:
                print(f"  • '{word}' already in vocab (ID: {self.vocab[word]})")
            else:
                self.vocab[word] = self.next_id
                self.idx_to_word[self.next_id] = word
                print(f"  ✓ Added '{word}' with ID: {self.next_id}")
                self.next_id += 1
                added_count += 1
        
        return added_count
    
    def save(self, output_path: str = None):
        """
        Save updated vocabulary.
        
        Args:
            output_path: Optional different path (default: overwrites original)
        """
        if output_path is None:
            output_path = self.vocab_path
        
        data = {
            'vocab': self.vocab,
            'idx_to_word': {str(k): v for k, v in self.idx_to_word.items()},
            'next_id': self.next_id
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Saved updated vocabulary to: {output_path}")
        print(f"  Total words: {len(self.vocab):,}")
    
    def get_stats(self):
        """Print vocabulary statistics"""
        print("\nVocabulary Statistics:")
        print(f"  Total words: {len(self.vocab):,}")
        print(f"  Next ID: {self.next_id}")
        print(f"  Path: {self.vocab_path}")


# ============================================================
# STANDALONE MODE - Add words here
# ============================================================

if __name__ == "__main__":
    
    # ========================================
    # CONFIGURATION
    # ========================================
    
    VOCAB_PATH = r'c:\Marina\models\corpus_ancora\corpus_ancoraC_vocab.json'
    
    # Words to add - EDIT THIS LIST
    NEW_WORDS = [
        "Lady",
        "Serendipity",
        # Add more words here as needed
    ]
    
    # Optional: Save to a different file (or None to overwrite)
    OUTPUT_PATH = None  # Set to a path like 'corpus_ancoraC_vocab_updated.json'
    
    # ========================================
    # UPDATE VOCABULARY
    # ========================================
    
    print("=" * 70)
    print("VOCABULARY UPDATER")
    print("=" * 70)
    
    try:
        updater = VocabUpdater(VOCAB_PATH)
    except FileNotFoundError:
        print(f"\n✗ ERROR: Could not find vocab file at:")
        print(f"  {VOCAB_PATH}")
        exit(1)
    
    print("\n" + "─" * 70)
    print("Adding new words...")
    print("─" * 70)
    
    added = updater.add_words(NEW_WORDS)
    
    print("\n" + "─" * 70)
    print(f"Added {added} new words")
    print("─" * 70)
    
    if added > 0:
        # Show before/after stats
        updater.get_stats()
        
        # Ask for confirmation
        print("\n" + "=" * 70)
        if OUTPUT_PATH:
            print(f"Save updated vocab to: {OUTPUT_PATH}")
        else:
            print(f"⚠ WARNING: This will OVERWRITE the original vocab file!")
            print(f"Original: {VOCAB_PATH}")
        
        confirm = input("\nProceed? (y/n): ").strip().lower()
        
        if confirm == 'y':
            updater.save(OUTPUT_PATH)
            print("\n✓ Update complete!")
            
            if OUTPUT_PATH is None:
                print("\n⚠ IMPORTANT: You must RETRAIN your model with the updated vocab!")
                print("   The model checkpoint still uses the old vocab size.")
                print("   Either:")
                print("   1. Retrain from scratch with new vocab")
                print("   2. Restore original vocab file")
        else:
            print("\n✗ Update cancelled.")
    else:
        print("\n✓ No changes needed - all words already in vocabulary.")
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
