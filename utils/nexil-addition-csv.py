# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 14:39:46 2025

@author: Kevin
"""

import pandas as pd
import re

# Define your mapping of patterns to their Nexil replacements
# We use word boundaries (\b) to avoid partial matches inside words
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
    r'\bWho can': 'Who-can',
}


def compress_questions(text):
    """
    Replaces multi-word question patterns with their single Nexil equivalents.
    """
    if pd.isna(text):
        return text
        
    compressed_text = str(text)
    for pattern, nexil in NEXIL_MAP.items():
        compressed_text = re.sub(pattern, nexil, compressed_text)
    return compressed_text

def main():
    # Read the CSV file
    input_file = 'solar_system.csv'  # Replace with your input file path
    output_file = 'solar_system3.csv'  # Replace with your desired output path
    
    df = pd.read_csv(input_file)
    
    # Apply the compression to the first column
    first_column_name = df.columns[0]
    df[first_column_name] = df[first_column_name].apply(compress_questions)
    
    # Save the modified DataFrame
    df.to_csv(output_file, index=False)
    print(f"Processing complete. Compressed data saved to: {output_file}")
    
    # Print a few examples to verify
    print("\nSample of changes:")
    sample_df = df.head(10)
    for _, row in sample_df.iterrows():
        original = row[first_column_name]
        print(f"  {original}")

if __name__ == "__main__":
    main()