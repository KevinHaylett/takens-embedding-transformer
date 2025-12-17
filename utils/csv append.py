# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 12:14:55 2025

@author: Kevin
"""

import csv
import sys

def prepend_to_first_column(input_file, output_file, text_to_prepend="Can you tell me "):
    """
    Prepend text to the first column of a CSV file (excluding header).
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        text_to_prepend: Text to prepend to first column values
    """
    try:
        with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
             open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            
            # Write header unchanged
            header = next(reader)
            writer.writerow(header)
            
            # Process remaining rows
            for row in reader:
                if row:  # Skip empty rows
                    row[0] = text_to_prepend + row[0]
                    writer.writerow(row)
                    
        print(f"Successfully processed {input_file}")
        print(f"Output saved to {output_file}")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except Exception as e:
        print(f"Error: {e}")

# Usage example
if __name__ == "__main__":
    input_csv =  r'c:\Marina\models\solar_system\solar_system.csv' # Change this to your input file
    output_csv =  r'c:\Marina\models\solar_system\solar_system_append.csv'  # Change this to your desired output file
    
    # To modify the file in place, use the same name for both:
    # prepend_to_first_column("data.csv", "data_modified.csv")
    
    prepend_to_first_column(input_csv, output_csv)