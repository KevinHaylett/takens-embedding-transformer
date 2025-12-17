"""
Simple Marina Test - No Arguments Needed
Just press Run in Spyder!
"""

import torch
import sys

# Add path to your uploaded files
sys.path.append('.')  # Current directory

from mvec_encoder import MVecEncoder
from mvec_model import MVecLanguageModel
from tbt_architecture import create_exponential_delays
#from nexil_create import to_nexils 
# ============================================================
# CONFIGURATION - Edit these as needed
# ============================================================

MODEL_PATH =  r'c:\Marina\models\solar_system\solar_system_model.pt'
VOCAB_PATH = r'c:\Marina\models\solar_system\solar_system_vocab.json'
DEVICE = 'cpu'  # or 'cuda' if you have GPU

# Test questions
TEST_QUESTIONS = [

"How hot is Mercury?",
"Does Mercury have Moons?",
"How long would it take to get to Venus?"

    ]
# One-liner conversion — now every question is Marina-ready
#TEST_QUESTIONS = [to_nexils(q) for q in TEST_QUESTIONS]


# Generation settings
MAX_TOKENS = 100
TEMPERATURE = 0.5#0.8
TOP_K = 50#50

# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("Marina Simple Test")
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

# Show training metrics
if 'val_metrics' in checkpoint:
    val = checkpoint['val_metrics']
    print(f"\nTraining metrics:")
    print(f"  Word loss: {val['word_loss']:.4f}")
    print(f"  End loss: {val['end_loss']:.4f}")

# ============================================================
# GENERATION FUNCTION
# ============================================================

def generate_answer(question, verbose=True):
    """Generate Marina's answer to a question"""
    
    # Encode question as USER
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
                print(f"{step:<6} {word:<15} {end_prob_yes:>6.1%}")
            
            # Update tensors
            input_ids = torch.cat([input_ids, torch.tensor([[next_word_id]], device=DEVICE)], dim=1)
            identity_ids = torch.cat([identity_ids, torch.tensor([[2]], device=DEVICE)], dim=1)
            
            # Check stopping
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
# RUN TESTS
# ============================================================

print("\n" + "=" * 70)
print("TESTING")
print("=" * 70)

for question in TEST_QUESTIONS:
    answer = generate_answer(question, verbose=True)

print("\n" + "=" * 70)
print("All tests complete!")
print("=" * 70)

# ============================================================
# INTERACTIVE LOOP (Optional)
# ============================================================

print("\nInteractive mode - type your question or 'quit' to exit:")

while True:
    try:
        user_input = input("\nYou: ").strip()
      
        if not user_input:
            continue
            
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        # This is the only change — convert natural question → nexil format
       ## marina_input = to_nexils(user_input)
        
        # Optional: show the user what Marina actually "hears" (great for debugging/demo)
        # print(f"→ Marina sees: {marina_input}")
        answer = generate_answer(user_input, verbose=False)
        ##answer = generate_answer(marina_input, verbose=False)
        print(f"Marina: {answer}")
        
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
    except EOFError:
        # Spyder / some IDEs send EOF on stop
        print("\nSession ended")
        break
