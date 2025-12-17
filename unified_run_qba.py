"""
Marina QBA Test - Two-Phase Generation
Question → Bridge (INTERNAL) → Answer (VISIBLE)

The bridge is generated but hidden - it's the geometric pathway through meaning-space!
"""

import torch
import sys

sys.path.append('.')

from mvec_encoder import MVecEncoder
from mvec_model import MVecLanguageModel
from tbt_architecture import create_exponential_delays

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r'c:\Marina\models\qba\qba_model.pt'
VOCAB_PATH = r'c:\Marina\models\qba\qba_vocab.json'
DEVICE = 'cpu'

# Test questions
TEST_QUESTIONS = [
    "What is the capital of France?",
    "What is the boiling point of water?",
]

# Generation settings
MAX_BRIDGE_TOKENS = 20   # Bridge should be short
MAX_ANSWER_TOKENS = 100  # Answer can be longer
TEMPERATURE = 0.8
TOP_K = 50

# Display options
SHOW_BRIDGE = True  # Set to False to hide bridge (like production mode)

# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("Marina QBA Test - Two-Phase Generation")
print("=" * 70)

print("\nLoading model...")
encoder = MVecEncoder()
encoder.load_vocab(VOCAB_PATH)
print(f"Vocabulary: {encoder.get_vocab_size()} words")

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

print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

if 'val_metrics' in checkpoint:
    val = checkpoint['val_metrics']
    print(f"\nTraining metrics:")
    print(f"  Word loss: {val['word_loss']:.4f}")
    print(f"  End loss: {val['end_loss']:.4f}")

print(f"\nMode: Two-phase QBA generation")
print(f"  Phase 1: Generate bridge (INTERNAL) - hidden reasoning")
print(f"  Phase 2: Generate answer (VISIBLE) - user sees this")

# ============================================================
# TWO-PHASE GENERATION FUNCTION
# ============================================================

def generate_answer_qba(question, verbose=True):
    """
    Generate answer using QBA two-phase approach:
    1. Generate bridge (INTERNAL) - the hidden geometric pathway
    2. Generate answer (VISIBLE) - the final output
    
    Returns: (bridge, answer) tuple
    """
    
    # Encode question as USER
    q_words, q_identity, q_end = encoder.encode_sequence(
        question, identity="USER", is_last_in_turn=True
    )
    
    input_ids = torch.tensor([q_words], dtype=torch.long).to(DEVICE)
    identity_ids = torch.tensor([q_identity], dtype=torch.long).to(DEVICE)
    
    bridge_tokens = []
    answer_tokens = []
    
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Question: {question}")
        print(f"{'=' * 70}")
    
    with torch.no_grad():
        # ================================================================
        # PHASE 1: Generate INTERNAL (Bridge) - Hidden Reasoning
        # ================================================================
        if verbose and SHOW_BRIDGE:
            print(f"\n🌉 BRIDGE (INTERNAL - Hidden Reasoning):")
            print(f"{'Step':<6} {'Word':<20} {'End Prob':<10}")
            print("-" * 45)
        
        for step in range(MAX_BRIDGE_TOKENS):
            # Forward pass
            word_logits, end_logits, _, _ = model(input_ids, identity_ids)
            
            # Sample next word
            next_word_logits = word_logits[:, -1, :] / TEMPERATURE
            
            if TOP_K > 0:
                v, _ = torch.topk(next_word_logits, min(TOP_K, next_word_logits.size(-1)))
                next_word_logits[next_word_logits < v[:, [-1]]] = -float('inf')
            
            probs = torch.softmax(next_word_logits, dim=-1)
            next_word_id = torch.multinomial(probs, num_samples=1).item()
            
            # Get end probability
            end_probs = torch.softmax(end_logits[:, -1, :], dim=-1)
            end_prob_yes = end_probs[0, 1].item()
            
            word = encoder.decode_word(next_word_id)
            bridge_tokens.append(next_word_id)
            
            if verbose and SHOW_BRIDGE:
                print(f"{step:<6} {word:<20} {end_prob_yes:>6.1%}")
            
            # Update with INTERNAL identity (1)
            input_ids = torch.cat([input_ids, torch.tensor([[next_word_id]], device=DEVICE)], dim=1)
            identity_ids = torch.cat([identity_ids, torch.tensor([[1]], device=DEVICE)], dim=1)
            
            # Stop if model signals end of bridge
            if end_prob_yes > 0.5:
                if verbose and SHOW_BRIDGE:
                    print(f"Bridge complete ({step+1} tokens)")
                break
        
        # ================================================================
        # PHASE 2: Generate VISIBLE (Answer) - What User Sees
        # ================================================================
        if verbose:
            print(f"\n💬 ANSWER (VISIBLE - User Output):")
            print(f"{'Step':<6} {'Word':<20} {'End Prob':<10}")
            print("-" * 45)
        
        for step in range(MAX_ANSWER_TOKENS):
            # Forward pass
            word_logits, end_logits, _, _ = model(input_ids, identity_ids)
            
            # Sample next word
            next_word_logits = word_logits[:, -1, :] / TEMPERATURE
            
            if TOP_K > 0:
                v, _ = torch.topk(next_word_logits, min(TOP_K, next_word_logits.size(-1)))
                next_word_logits[next_word_logits < v[:, [-1]]] = -float('inf')
            
            probs = torch.softmax(next_word_logits, dim=-1)
            next_word_id = torch.multinomial(probs, num_samples=1).item()
            
            # Get end probability
            end_probs = torch.softmax(end_logits[:, -1, :], dim=-1)
            end_prob_yes = end_probs[0, 1].item()
            
            word = encoder.decode_word(next_word_id)
            answer_tokens.append(next_word_id)
            
            if verbose:
                print(f"{step:<6} {word:<20} {end_prob_yes:>6.1%}")
            
            # Update with VISIBLE identity (2)
            input_ids = torch.cat([input_ids, torch.tensor([[next_word_id]], device=DEVICE)], dim=1)
            identity_ids = torch.cat([identity_ids, torch.tensor([[2]], device=DEVICE)], dim=1)
            
            # Stop if model signals end of answer
            if end_prob_yes > 0.5:
                if verbose:
                    print(f"Answer complete ({step+1} tokens)")
                break
    
    # Decode sequences
    bridge = encoder.decode_sequence(bridge_tokens)
    answer = encoder.decode_sequence(answer_tokens)
    
    if verbose:
        print(f"\n{'=' * 70}")
        if SHOW_BRIDGE:
            print(f"🌉 Bridge (hidden): {bridge}")
        print(f"💬 Answer: {answer}")
        print(f"{'=' * 70}")
    
    return bridge, answer


# ============================================================
# RUN TESTS
# ============================================================

print("\n" + "=" * 70)
print("TESTING - Two-Phase QBA Generation")
print("=" * 70)

results = []
for question in TEST_QUESTIONS:
    bridge, answer = generate_answer_qba(question, verbose=True)
    results.append({
        'question': question,
        'bridge': bridge,
        'answer': answer
    })

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

for i, result in enumerate(results, 1):
    print(f"\n{i}. Q: {result['question']}")
    if SHOW_BRIDGE:
        print(f"   Bridge: {result['bridge']}")
    print(f"   A: {result['answer']}")

print("\n" + "=" * 70)
print("All tests complete!")
print("=" * 70)

# ============================================================
# INTERACTIVE MODE
# ============================================================

print("\n" + "=" * 70)
print("Interactive Mode")
print("=" * 70)
print("Type your question or 'quit' to exit")
print("Set SHOW_BRIDGE=True in config to see hidden reasoning")
print("=" * 70)

while True:
    try:
        user_input = input("\nYou: ").strip()
        
        if not user_input:
            continue
            
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        # Generate with two-phase approach
        bridge, answer = generate_answer_qba(user_input, verbose=False)
        
        # In production mode, only show answer
        # In debug mode, can show bridge too
        if SHOW_BRIDGE:
            print(f"Marina [bridge]: {bridge}")
        print(f"Marina: {answer}")
        
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
    except EOFError:
        print("\nSession ended")
        break

print("\n" + "=" * 70)
print("ABOUT QBA MODE")
print("=" * 70)
print("""
The Question-Bridge-Answer (QBA) architecture uses two phases:

Phase 1 - INTERNAL (Bridge):
  • Generated but hidden from user
  • Creates a geometric pathway through meaning-space
  • Acts as semantic coordinates for reasoning
  • Like scaffolding: necessary during construction, removed after

Phase 2 - VISIBLE (Answer):
  • Generated using the bridge as context
  • This is what the user sees
  • Flows naturally from the hidden reasoning

This is pure Geofinitism:
  • The bridge is a finite geometric path
  • Not arbitrary - learned from training data
  • A measurement-first trajectory through embedding space
  • Manifold geometry in action!

Research possibilities:
  • Probe bridges to see model's "reasoning"
  • Compare bridges for similar questions
  • Visualize question → bridge → answer trajectories
  • Test if consistent questions produce consistent bridges
  • Manually inject bridges to test causal influence
""")
print("=" * 70)
