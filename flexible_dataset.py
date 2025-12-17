"""
Flexible Dataset for Marina Training
 - flexible_dataset.py
====================================
Handles both CSV and plain text files with multiple chunking strategies.
"""

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import csv
from typing import List, Tuple, Optional
import re


class FlexibleMVecDataset(Dataset):
    """
    Dataset that handles multiple data formats and chunking strategies.
    
    Supported formats:
    - CSV with question,answer columns (QA mode)
    - CSV with question,bridge,answer columns (QBA mode)
    - CSV with text column (continuous mode)
    - Plain text files (continuous mode)
    
    Chunking modes:
    - 'pairs': Individual Q&A pairs (for CSV)
    - 'conversation': Multiple Q&A pairs chained together (for CSV)
    - 'sliding': Sliding windows with overlap (for text)
    - 'paragraph': Paragraph-based chunks (for text)
    - 'document': Full document (for text)
    """
    
    def __init__(
        self,
        data_path: str,
        encoder,
        mode: str = 'sliding',
        data_format: str = 'text',
        max_seq_len: int = 256,
        turns_per_conversation: int = 3,
        stride: int = 128,
        min_seq_len: int = 10
    ):
        self.data_path = data_path
        self.encoder = encoder
        self.mode = mode.lower()
        self.data_format = data_format.lower()
        self.max_seq_len = max_seq_len
        self.turns_per_conversation = turns_per_conversation
        self.stride = stride
        self.min_seq_len = min_seq_len
        
        # Storage for processed sequences
        self.sequences = []
        
        print(f"FlexibleMVecDataset Configuration:")
        print(f"  Mode: {self.mode}")
        print(f"  Format: {self.data_format}")
        print(f"  Max sequence length: {self.max_seq_len}")
        
        # Load and process data
        if self.data_format == 'csv':
            self._load_csv()
        else:
            self._load_text()
        
        print(f"  Created {len(self.sequences)} training sequences")
    
    def _load_csv(self):
        """Load and process CSV data"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = [h.lower() for h in reader.fieldnames]
            
            # Detect CSV format
            has_bridge = 'bridge' in headers
            has_text = 'text' in headers
            
            rows = list(reader)
            
            if has_text:
                # CSV with continuous text
                self._process_csv_text(rows)
            elif has_bridge:
                # QBA format
                self._process_csv_qba(rows)
            else:
                # QA format
                self._process_csv_qa(rows)
    
    def _process_csv_qa(self, rows):
        """Process CSV with question,answer format"""
        qa_pairs = []
        
        for row in rows:
            q = row.get('question') or row.get('user', '')
            a = row.get('answer') or row.get('response', '')
            
            if q.strip() and a.strip():
                qa_pairs.append((q, a))
        
        if self.mode == 'pairs':
            # Individual pairs
            for q, a in qa_pairs:
                # encode_qa returns (word_ids, identity_ids, end_ids)
                word_ids, identity_ids, end_ids = self.encoder.encode_qa(q, a)
                
                if len(word_ids) >= self.min_seq_len:
                    # Truncate if needed
                    if len(word_ids) > self.max_seq_len:
                        word_ids = word_ids[:self.max_seq_len]
                        identity_ids = identity_ids[:self.max_seq_len]
                        end_ids = end_ids[:self.max_seq_len]
                    
                    self.sequences.append((word_ids, identity_ids, end_ids))
        
        elif self.mode == 'conversation':
            # Chain multiple pairs
            for i in range(0, len(qa_pairs), self.turns_per_conversation):
                chunk = qa_pairs[i:i + self.turns_per_conversation]
                if chunk:
                    # Combine multiple Q&A pairs
                    combined_word_ids = []
                    combined_identity_ids = []
                    combined_end_ids = []
                    
                    for q, a in chunk:
                        # encode_qa returns (word_ids, identity_ids, end_ids)
                        word_ids, identity_ids, end_ids = self.encoder.encode_qa(q, a)
                        combined_word_ids.extend(word_ids)
                        combined_identity_ids.extend(identity_ids)
                        combined_end_ids.extend(end_ids)
                    
                    # Truncate if needed
                    if len(combined_word_ids) > self.max_seq_len:
                        combined_word_ids = combined_word_ids[:self.max_seq_len]
                        combined_identity_ids = combined_identity_ids[:self.max_seq_len]
                        combined_end_ids = combined_end_ids[:self.max_seq_len]
                    
                    if len(combined_word_ids) >= self.min_seq_len:
                        self.sequences.append((
                            combined_word_ids,
                            combined_identity_ids,
                            combined_end_ids
                        ))
        else:
            # Default to pairs for unsupported modes
            for q, a in qa_pairs:
                # encode_qa returns (word_ids, identity_ids, end_ids)
                word_ids, identity_ids, end_ids = self.encoder.encode_qa(q, a)
                
                if len(word_ids) >= self.min_seq_len:
                    # Truncate if needed
                    if len(word_ids) > self.max_seq_len:
                        word_ids = word_ids[:self.max_seq_len]
                        identity_ids = identity_ids[:self.max_seq_len]
                        end_ids = end_ids[:self.max_seq_len]
                    
                    self.sequences.append((word_ids, identity_ids, end_ids))
    
    def _process_csv_qba(self, rows):
        """Process CSV with question,bridge,answer format"""
        qba_triples = []
        
        for row in rows:
            q = row.get('question') or row.get('user', '')
            b = row.get('bridge', '')
            a = row.get('answer') or row.get('response', '')
            
            if q.strip() and a.strip():
                qba_triples.append((q, b, a))
        
        if self.mode == 'pairs':
            # Individual triples
            for q, b, a in qba_triples:
                # encode_qba returns (word_ids, identity_ids, end_ids)
                word_ids, identity_ids, end_ids = self.encoder.encode_qba(q, b, a)
                
                if len(word_ids) >= self.min_seq_len:
                    # Truncate if needed
                    if len(word_ids) > self.max_seq_len:
                        word_ids = word_ids[:self.max_seq_len]
                        identity_ids = identity_ids[:self.max_seq_len]
                        end_ids = end_ids[:self.max_seq_len]
                    
                    self.sequences.append((word_ids, identity_ids, end_ids))
        
        elif self.mode == 'conversation':
            # Chain multiple triples
            for i in range(0, len(qba_triples), self.turns_per_conversation):
                chunk = qba_triples[i:i + self.turns_per_conversation]
                if chunk:
                    combined_word_ids = []
                    combined_identity_ids = []
                    combined_end_ids = []
                    
                    for q, b, a in chunk:
                        # encode_qba returns (word_ids, identity_ids, end_ids)
                        word_ids, identity_ids, end_ids = self.encoder.encode_qba(q, b, a)
                        combined_word_ids.extend(word_ids)
                        combined_identity_ids.extend(identity_ids)
                        combined_end_ids.extend(end_ids)
                    
                    if len(combined_word_ids) > self.max_seq_len:
                        combined_word_ids = combined_word_ids[:self.max_seq_len]
                        combined_identity_ids = combined_identity_ids[:self.max_seq_len]
                        combined_end_ids = combined_end_ids[:self.max_seq_len]
                    
                    if len(combined_word_ids) >= self.min_seq_len:
                        self.sequences.append((
                            combined_word_ids,
                            combined_identity_ids,
                            combined_end_ids
                        ))
        else:
            # Default to pairs
            for q, b, a in qba_triples:
                # encode_qba returns (word_ids, identity_ids, end_ids)
                word_ids, identity_ids, end_ids = self.encoder.encode_qba(q, b, a)
                
                if len(word_ids) >= self.min_seq_len:
                    # Truncate if needed
                    if len(word_ids) > self.max_seq_len:
                        word_ids = word_ids[:self.max_seq_len]
                        identity_ids = identity_ids[:self.max_seq_len]
                        end_ids = end_ids[:self.max_seq_len]
                    
                    self.sequences.append((word_ids, identity_ids, end_ids))
    
    def _process_csv_text(self, rows):
        """Process CSV with text column"""
        texts = [row.get('text', '') for row in rows]
        texts = [t.strip() for t in texts if t.strip()]
        
        # Join all texts and process as continuous text
        full_text = ' '.join(texts)
        self._process_continuous_text(full_text)
    
    def _load_text(self):
        """Load and process plain text file"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        self._process_continuous_text(text)
    
    def _process_continuous_text(self, text: str):
        """Process continuous text with various chunking strategies"""
        
        if self.mode == 'sliding':
            self._sliding_window_chunks(text)
        
        elif self.mode == 'paragraph':
            self._paragraph_chunks(text)
        
        elif self.mode == 'document':
            self._document_chunks(text)
        
        else:
            # Default to sliding
            print(f"  Warning: mode '{self.mode}' not ideal for text, using sliding")
            self._sliding_window_chunks(text)
    
    def _sliding_window_chunks(self, text: str):
        """Create overlapping chunks with sliding window"""
        # Tokenize the entire text first
        words = text.split()
        
        if len(words) < self.min_seq_len:
            print(f"  Warning: Text too short ({len(words)} words < {self.min_seq_len} min)")
            return
        
        # Create sliding windows at word level
        start = 0
        num_chunks = 0
        while start < len(words):
            end = min(start + self.max_seq_len, len(words))
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            
            if len(chunk_words) >= self.min_seq_len:
                # Encode this chunk as continuous text
                # Note: encode_continuous uses 'chunk_size' parameter, not 'max_len'
                try:
                    seq = self.encoder.encode_continuous(chunk_text)
                    if seq and len(seq[0]) >= self.min_seq_len:
                        # Truncate if needed
                        if len(seq[0]) > self.max_seq_len:
                            seq = (
                                seq[0][:self.max_seq_len],
                                seq[1][:self.max_seq_len],
                                seq[2][:self.max_seq_len]
                            )
                        self.sequences.append(seq)
                        num_chunks += 1
                except Exception as e:
                    print(f"  Warning: Failed to encode chunk: {e}")
            
            # Move forward by stride
            start += self.stride
            
            # Stop if we've reached the end
            if end >= len(words):
                break
        
        if num_chunks == 0:
            print(f"  Warning: No valid chunks created from text")
    
    def _paragraph_chunks(self, text: str):
        """Create chunks based on paragraph boundaries"""
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        for para in paragraphs:
            words = para.split()
            
            if len(words) < self.min_seq_len:
                continue
            
            # If paragraph is too long, split it
            if len(words) > self.max_seq_len:
                # Use sliding window within this paragraph
                start = 0
                while start < len(words):
                    end = min(start + self.max_seq_len, len(words))
                    chunk_words = words[start:end]
                    chunk_text = ' '.join(chunk_words)
                    
                    if len(chunk_words) >= self.min_seq_len:
                        try:
                            seq = self.encoder.encode_continuous(chunk_text)
                            if seq and len(seq[0]) >= self.min_seq_len:
                                # Truncate if needed
                                if len(seq[0]) > self.max_seq_len:
                                    seq = (
                                        seq[0][:self.max_seq_len],
                                        seq[1][:self.max_seq_len],
                                        seq[2][:self.max_seq_len]
                                    )
                                self.sequences.append(seq)
                        except Exception as e:
                            print(f"  Warning: Failed to encode paragraph chunk: {e}")
                    
                    start += self.stride
                    if end >= len(words):
                        break
            else:
                # Paragraph fits in one chunk
                try:
                    seq = self.encoder.encode_continuous(para)
                    if seq and len(seq[0]) >= self.min_seq_len:
                        # Truncate if needed
                        if len(seq[0]) > self.max_seq_len:
                            seq = (
                                seq[0][:self.max_seq_len],
                                seq[1][:self.max_seq_len],
                                seq[2][:self.max_seq_len]
                            )
                        self.sequences.append(seq)
                except Exception as e:
                    print(f"  Warning: Failed to encode paragraph: {e}")
    
    def _document_chunks(self, text: str):
        """Process entire document as one or more max-length chunks"""
        words = text.split()
        
        if len(words) < self.min_seq_len:
            print(f"  Warning: Text too short ({len(words)} words)")
            return
        
        # Split into max_seq_len chunks at word level
        for start in range(0, len(words), self.max_seq_len):
            end = min(start + self.max_seq_len, len(words))
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            
            if len(chunk_words) >= self.min_seq_len:
                try:
                    seq = self.encoder.encode_continuous(chunk_text)
                    if seq and len(seq[0]) >= self.min_seq_len:
                        # Truncate if needed
                        if len(seq[0]) > self.max_seq_len:
                            seq = (
                                seq[0][:self.max_seq_len],
                                seq[1][:self.max_seq_len],
                                seq[2][:self.max_seq_len]
                            )
                        self.sequences.append(seq)
                except Exception as e:
                    print(f"  Warning: Failed to encode document chunk: {e}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        word_ids, identity_ids, end_ids = self.sequences[idx]
        
        # Convert to tensors
        word_ids_tensor = torch.tensor(word_ids, dtype=torch.long)
        identity_tensor = torch.tensor(identity_ids, dtype=torch.long)
        end_ids_tensor = torch.tensor(end_ids, dtype=torch.long)
        
        # For autoregressive training: input = sequence[:-1], target = sequence[1:]
        # But we return full sequence and let trainer handle the shift
        return word_ids_tensor, identity_tensor, end_ids_tensor


def collate_flexible_batch(batch):
    """
    Collate function for flexible batching.
    Pads sequences and creates input/target pairs for autoregressive training.
    
    CRITICAL: END markers must align with TARGET tokens, not input tokens!
    
    Returns:
        input_ids: [batch, seq_len-1] - input tokens (all but last)
        target_ids: [batch, seq_len-1] - target tokens (all but first)
        identity_ids: [batch, seq_len-1] - identity channel (shifted with input)
        end_ids: [batch, seq_len-1] - END markers (shifted with TARGET, not input!)
    """
    word_ids_list, identity_list, end_ids_list = zip(*batch)
    
    # Pad sequences
    word_ids_padded = pad_sequence(word_ids_list, batch_first=True, padding_value=0)
    identity_padded = pad_sequence(identity_list, batch_first=True, padding_value=0)
    end_ids_padded = pad_sequence(end_ids_list, batch_first=True, padding_value=0)
    
    # Create input/target pairs for autoregressive training
    # input_ids: all tokens except the last one
    # target_ids: all tokens except the first one (shifted by 1)
    input_ids = word_ids_padded[:, :-1]
    target_ids = word_ids_padded[:, 1:]
    
    # Identity markers should align with INPUT (what channel we're in)
    identity_ids = identity_padded[:, :-1]
    
    # END markers must align with TARGET (what we're predicting)
    # NOT shifted with input, but shifted with target!
    end_ids = end_ids_padded[:, 1:]  # Same shift as target_ids
    
    return input_ids, target_ids, identity_ids, end_ids
