# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 18:27:31 2025

@author: Kevin
"""

import re

# ------------------------------------------------------------------
# Your full nexil map (exactly as you posted – kept unchanged)
# ------------------------------------------------------------------
NEXIL_MAP = {
    r'\bWhat is that\b': 'What-is-that',
    r'\bWhat is the\b': 'What-is-the',
    r'\bWhat is this\b': 'What-is-this',
    r'\bWhat is it\b': 'What-is-it',
    r'\bWhat is\b': 'What-is',
    r'\bWhat did\b': 'What-did',
    r'\bWhat does\b': 'What-does',
    r'\bWhat are\b': 'What-are',
    r'\bWhat type\b': 'What-type',
    r'\bWhat do\b': 'What-do',
    r'\bWhat have\b': 'What-have',
    r'\bWhat was\b': 'What-was',
    r'\bWhat made\b': 'What-made',
    r'\bWhat were\b': 'What-were',
    r'\bWhat kind\b': 'What-kind',
    
    r'\bHow are they\b': 'How-are-they',
    r'\bHow is that\b': 'How-is-that',
    r'\bHow is this\b': 'How-is-this',
    r'\bHow is the\b': 'How-is-the',
    r'\bHow is it\b': 'How-is-it',
    r'\bHow is\b': 'How-is',
    r'\bHow did\b': 'How-did',
    r'\bHow would\b': 'How-would',
    r'\bHow does\b': 'How-does',
    r'\bHow are\b': 'How-are',
    r'\bHow might\b': 'How-might',
    r'\bHow many\b': 'How-many',
    
    r'\bWhy are they\b': 'Why-are-they',
    r'\bWhy is this\b': 'Why-is-this',
    r'\bWhy is it\b': 'Why-is-it',
    r'\bWhy is\b': 'Why-is',
    r'\bWhy are\b': 'Why-are',
    r'\bWhy would\b': 'Why-would',
    r'\bWhy did\b': 'Why-did',
    r'\bWhy does\b': 'Why-does',
    r'\bWhy might\b': 'Why-might',
    
    r'\bWhere can they\b': 'Where-can-they',
    r'\bWhere can it\b': 'Where-can-it',
    r'\bWhere can\b': 'Where-can',
    r'\bWhere can I\b': 'Where-can-I',
    r'\bWhere can we\b': 'Where-can-we',
    r'\bWhere is\b': 'Where-is',
    r'\bWhere are\b': 'Where-are',
    
    r'\bWhen is it\b': 'When-is-it',
    r'\bWhen does\b': 'When-does',
    r'\bWhen are they\b': 'When-are-they',
    r'\bWhen are\b': 'When-are',
    
    r'\bCan it\b': 'can-it',
    r'\bCan they\b': 'Can-they',
    r'\bCan you\b': 'Can-you',
    
    r'\bWill it\b': 'Will-it',
    r'\bWill they\b': 'Will-they',
    r'\bWill you\b': 'Will-you',
    
    r'\bWho is this\b': 'Who-is-this',
    r'\bWho are they\b': 'Who-are-they',
    r'\bWho was\b': 'Who-was',
    r'\bWho is\b': 'Who-is',
    r'\bWho did\b': 'Who-did',
    r'\bWho were\b': 'Who-were',
    r'\bWho has\b': 'Who-has',
    r'\bWho had\b': 'Who-had',
    r'\bWho are\b': 'Who-are',
    r'\bWho can\b': 'Who-can',
}

# Sort by length descending so longer patterns match first (critical!)
NEXIL_RULES = sorted(NEXIL_MAP.items(), key=lambda x: len(x[0]), reverse=True)

def to_nexils(prompt: str) -> str:
    """
    Convert a natural-language question/prompt into the nexil format
    expected by Marina during inference.
    
    Example:
        "What is the definition of inductive inference?"
        → "What-is-the definition of inductive inference?"
    """
    text = prompt.strip()
    for pattern, replacement in NEXIL_RULES:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

# ------------------------------------------------------------------
# Quick test (uncomment to try)
# ------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        "What is the role of dharmas in Abhidharma?",
        "Why does the doctrine of momentariness create problems?",
        "How is the self understood in early Buddhism?",
        "Can you explain the difference between general and specific ability?",
        "Who is considered the founder of the Abhidharma tradition?"
    ]
    
    print("Nexil conversion demo:\n")
    for t in tests:
        print(f"Original : {t}")
        print(f"Nexil    : {to_nexils(t)}\n")