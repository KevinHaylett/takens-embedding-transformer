"""
Example: Integrating VocabChecker into unified_run.py

This shows how to add vocab checking before inference to catch OOV words early.
"""

import torch
import sys

sys.path.append('.')

from mvec_encoder import MVecEncoder
from mvec_model import MVecLanguageModel
from tbt_architecture import create_exponential_delays
from vocab_checker import VocabChecker  # <-- NEW IMPORT

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r'c:\Marina\models\corpus_ancora_2\corpus_ancora_2_model.pt'
VOCAB_PATH = r'c:\Marina\models\corpus_ancora_2\corpus_ancora_2_vocab.json'
DEVICE = 'cpu'

TEST_QUESTIONS = [
    "Where is Lady Serendipity?",  # Has OOV words
    "How do I find Kaevin the Listener?",  # All in vocab
    "Where are the mice in the walls?",  # All in vocab
]

MAX_TOKENS = 100
TEMPERATURE = 0.8
TOP_K = 50

# ============================================================
# LOAD MODEL (same as before)
# ============================================================

print("=" * 70)
print("Marina with Vocab Checking")
print("=" * 70)

encoder = MVecEncoder()
encoder.load_vocab(VOCAB_PATH)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
config = checkpoint['model_config']

model = MVecLanguageModel(
    vocab_size=config['vocab_size'],
    embed_dim=config['embed_dim'],
    hidden_dim=config['hidden_dim'],
    num_layers=config['num_layers'],
    max_seq_len=512,
    delays=create_exponential_delays(128),
    dropout=0.1,
    use_identity_embed=config['use_identity_embed'],
    identity_embed_dim=config['identity_embed_dim'],
    tie_weights=False
)

model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

# ============================================================
# NEW: Initialize VocabChecker
# ============================================================

vocab_checker = VocabChecker(VOCAB_PATH)

print("\n" + "=" * 70)
print("PRE-FLIGHT VOCABULARY CHECK")
print("=" * 70)

# Check all test questions BEFORE trying to run them
results = vocab_checker.check_questions(TEST_QUESTIONS, verbose=True)

# ============================================================
# GENERATION FUNCTION (same as before)
# ============================================================

def generate_answer(question, verbose=True):
    """Generate Marina's answer to a question"""
    
    q_words, q_identity, q_end = encoder.encode_sequence(
        question, identity="USER", is_last_in_turn=True
    )
    
    input_ids = torch.tensor([q_words], dtype=torch.long).to(DEVICE)
    identity_ids = torch.tensor([q_identity], dtype=torch.long).to(DEVICE)
    
    answer_tokens = []
    
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Question: {question}")
        print(f"{'=' * 70}")
        print(f"\n{'Step':<6} {'Word':<15} {'End Prob':<10}")
        print("-" * 40)
    
    with torch.no_grad():
        for step in range(MAX_TOKENS):
            word_logits, end_logits, _, _ = model(input_ids, identity_ids)
            
            next_word_logits = word_logits[:, -1, :] / TEMPERATURE
            
            if TOP_K > 0:
                v, _ = torch.topk(next_word_logits, min(TOP_K, next_word_logits.size(-1)))
                next_word_logits[next_word_logits < v[:, [-1]]] = -float('inf')
            
            probs = torch.softmax(next_word_logits, dim=-1)
            next_word_id = torch.multinomial(probs, num_samples=1).item()
            
            end_probs = torch.softmax(end_logits[:, -1, :], dim=-1)
            end_prob_yes = end_probs[0, 1].item()
            
            word = encoder.decode_word(next_word_id)
            answer_tokens.append(next_word_id)
            
            if verbose:
                print(f"{step:<6} {word:<15} {end_prob_yes:>6.1%}")
            
            input_ids = torch.cat([input_ids, torch.tensor([[next_word_id]], device=DEVICE)], dim=1)
            identity_ids = torch.cat([identity_ids, torch.tensor([[2]], device=DEVICE)], dim=1)
            
            if end_prob_yes > 0.5:
                if verbose:
                    print(f"\nStopped after {step+1} tokens (end_prob={end_prob_yes:.1%})")
                break
    
    answer = encoder.decode_sequence(answer_tokens)
    
    if verbose:
        print(f"\nAnswer: {answer}")
        print("=" * 70)
    
    return answer

# ============================================================
# RUN TESTS WITH VOCAB CHECKING
# ============================================================

print("\n" + "=" * 70)
print("TESTING WITH VOCAB SAFETY CHECKS")
print("=" * 70)

for i, question in enumerate(TEST_QUESTIONS):
    
    # NEW: Check vocab BEFORE inference
    is_safe = vocab_checker.is_safe_for_inference(question)
    
    if not is_safe:
        print(f"\n⚠ WARNING: Question {i+1} contains OOV words!")
        _, oov_words, _ = vocab_checker.check_text(question, verbose=False)
        print(f"   OOV words: {', '.join(set(oov_words))}")
        print(f"   This will produce <unk> tokens in the output.")
        
        # Show suggestions
        print(f"\n   Suggestions:")
        for word in set(oov_words):
            suggestions = vocab_checker.suggest_alternatives(word, max_suggestions=3)
            if suggestions:
                print(f"     '{word}' → {', '.join(f'{w}' for w, _ in suggestions)}")
        
        # DECISION: Skip this question or proceed anyway?
        # Option 1: Skip
        print(f"\n   Skipping this question due to OOV words.\n")
        continue
        
        # Option 2: Proceed anyway (comment out the continue above)
        # print(f"\n   Proceeding anyway (will generate <unk> tokens)...\n")
    
    # If we get here, question is safe (or we chose to proceed anyway)
    answer = generate_answer(question, verbose=True)

print("\n" + "=" * 70)
print("All tests complete!")
print("=" * 70)

# ============================================================
# INTERACTIVE LOOP WITH VOCAB CHECKING
# ============================================================

print("\nInteractive mode with vocab checking - type 'quit' to exit:")

while True:
    try:
        user_input = input("\nYou: ").strip()
        
        if not user_input:
            continue
            
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        # NEW: Check vocab before inference
        if not vocab_checker.is_safe_for_inference(user_input):
            print("\n⚠ Warning: Your question contains unknown words:")
            _, oov_words, stats = vocab_checker.check_text(user_input, verbose=False)
            print(f"   OOV: {', '.join(set(oov_words))}")
            print(f"   Coverage: {stats['coverage_pct']:.1f}%")
            
            # Ask user if they want to proceed
            proceed = input("   Proceed anyway? (y/n): ").strip().lower()
            if proceed != 'y':
                print("   Skipping this question.")
                continue
        
        # Generate answer
        answer = generate_answer(user_input, verbose=False)
        print(f"Marina: {answer}")
        
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
    except EOFError:
        print("\nSession ended")
        break
