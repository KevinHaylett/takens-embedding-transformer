# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 21:53:06 2025

@author: Kevin
"""

import random
import re
import csv

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
INPUT_FILE = 'solar_system.csv'  # Replace with your actual file name
OUTPUT_FILE = 'solar_system2_aug.csv'
AUGMENTATION_FACTOR = 3  # How many new versions to create per line

# ---------------------------------------------------------
# AUGMENTATION TEMPLATES
# ---------------------------------------------------------

rules = [
    # "What is..." patterns
    (r"^What-is (.*?)\??$", [
        "Define {}.",
        "Tell me about {}.",
        "Describe {}.",
        "Can you explain what {} is?",
        "what is {}?"
    ]),
    
    # "Why is..." patterns
    (r"^Why-is (.*?)\??$", [
        "Explain why {}.",
        "What-is-the reason {}?",
        "How come {}?",
        "Tell me the reason {}."
    ]),
    
    # "Why does..." patterns
    (r"^Why-does (.*?)\??$", [
        "Explain why {}.",
        "What-is-the reason {}?",
        "How come {}?",
        "Tell me the reason {}."
    ]),
    
    # "How does..." patterns
    (r"^How-does (.*?)\??$", [
        "In what way {}?",
        "Explain the process of how {}."
    ]),
    
    # "Where is..." patterns
    (r"^Where-is (.*?)\??$", [
        "Location of {}?",
        "Tell me where to find {}.",
        "Where can I find {}?"
    ]),
    
    # "Who is..." patterns (added as example)
    (r"^Who-is (.*?)\??$", [
        "Tell me about {}.",
        "Can you describe {}?",
        "Who was {}?"
    ])
]

def augment_line(line):
    """
    Takes a single line of text. If it looks like a question,
    generates variations. Returns a list of strings (including original).
    """
    line = line.strip()
    if not line: 
        return []
    
    # Keep the original
    variations = {line} 
    
    # Check against all rules
    for pattern, templates in rules:
        match = re.match(pattern, line, re.IGNORECASE)
        if match:
            groups = match.groups()
            
            # Generate N random variations
            num_variations = min(len(templates), AUGMENTATION_FACTOR)
            selected_templates = random.sample(templates, num_variations)
            
            for temp in selected_templates:
                # Format the template with the captured groups
                try:
                    new_q = temp.format(*groups)
                except (IndexError, KeyError):
                    # Fallback if formatting fails
                    new_q = temp.format(groups[0] if groups else "")
                
                # Add question mark if missing
                if not new_q.endswith(('?', '.', '!')):
                    new_q += "?"
                    
                variations.add(new_q)
            
            # Once we match a rule, stop checking others
            break
    
    return list(variations)

# ---------------------------------------------------------
# MAIN EXECUTION - HANDLES CSV PROPERLY
# ---------------------------------------------------------
def main():
    print(f"Reading {INPUT_FILE}...")
    
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            # Try to read as CSV first
            try:
                csv_reader = csv.reader(f)
                rows = list(csv_reader)
                has_csv_format = len(rows) > 0 and len(rows[0]) >= 2
            except:
                has_csv_format = False
                f.seek(0)
                lines = f.readlines()
                
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE}. Please check the filename.")
        return

    total_original = 0
    total_augmented = 0
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as f_out:
        if has_csv_format:
            csv_writer = csv.writer(f_out)
            
            for row in rows:
                if len(row) >= 2:
                    question = row[0].strip()
                    answer = row[1].strip() if len(row) > 1 else ""
                    
                    # Augment the question only
                    augmented_questions = augment_line(question)
                    
                    for aug_q in augmented_questions:
                        csv_writer.writerow([aug_q, answer])
                        total_augmented += 1
                    
                    total_original += 1
                else:
                    # Write non-QA rows as-is
                    csv_writer.writerow(row)
        else:
            # Simple text file format
            for line in lines:
                line = line.strip()
                if "?" in line:  # Simple heuristic for questions
                    augmented_lines = augment_line(line)
                    for aug_line in augmented_lines:
                        f_out.write(aug_line + "\n")
                        total_augmented += 1
                    total_original += 1
                else:
                    f_out.write(line + "\n")

    print("------------------------------------------------")
    print(f"Done! Data expanded.")
    print(f"Original Questions: {total_original}")
    print(f"Total Lines Written: {total_augmented}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("------------------------------------------------")
    if has_csv_format:
        print("Format: CSV with Q,A pairs preserved")
    else:
        print("Format: Plain text (one question per line)")
    print("Next Step: Point your training script to the new augmented file.")

if __name__ == "__main__":
    main()