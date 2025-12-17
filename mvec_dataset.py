"""
MVec Dataset: mvec_dataset.py
Dataset class for multi-channel Marina training with mode support.
"""

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import csv
from typing import List, Tuple, Optional

from mvec_encoder import MVecEncoder, EncodingMode


class MVecDataset(Dataset):
    """
    Dataset for Marina multi-channel training.
    Supports QA, QBA, and CONTINUOUS encoding modes.
    """
    
    def __init__(
        self,
        csv_path: str,
        encoder: MVecEncoder,
        max_seq_len: int = 128,
        mode: EncodingMode = EncodingMode.QA,
        chunk_size: Optional[int] = None
    ):
        """
        Args:
            csv_path: Path to CSV file with training data
            encoder: MVecEncoder instance
            max_seq_len: Maximum sequence length
            mode: Encoding mode (QA, QBA, or CONTINUOUS)
            chunk_size: For CONTINUOUS mode, chunk size for END markers
        """
        self.encoder = encoder
        self.max_seq_len = max_seq_len
        self.mode = mode
        self.chunk_size = chunk_size
        self.data = []
        
        # Load data based on mode
        self._load_data(csv_path)
        
        print(f"Loaded {len(self.data)} samples in {mode.value.upper()} mode")
    
    def _load_data(self, csv_path: str):
        """Load data from CSV based on encoding mode"""
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                if self.mode == EncodingMode.QA:
                    # QA mode: question/user and answer/response columns
                    question = row.get('question') or row.get('user', '')
                    answer = row.get('answer') or row.get('response', '')
                    
                    if question and answer:
                        self.data.append({
                            'question': question,
                            'answer': answer
                        })
                
                elif self.mode == EncodingMode.QBA:
                    # QBA mode: question/user, bridge, answer/response columns
                    question = row.get('question') or row.get('user', '')
                    bridge = row.get('bridge', '')
                    answer = row.get('answer') or row.get('response', '')
                    
                    if question and bridge and answer:
                        self.data.append({
                            'question': question,
                            'bridge': bridge,
                            'answer': answer
                        })
                
                elif self.mode == EncodingMode.CONTINUOUS:
                    # CONTINUOUS mode: text or content column
                    text = row.get('text') or row.get('content', '')
                    
                    if text:
                        self.data.append({
                            'text': text
                        })
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a training sample.
        
        Returns:
            (input_ids, target_ids, identity_ids, end_ids)
        """
        item = self.data[idx]
        
        # Encode based on mode
        if self.mode == EncodingMode.QA:
            word_ids, identity_ids, end_ids = self.encoder.encode(
                EncodingMode.QA,
                question=item['question'],
                answer=item['answer']
            )
        
        elif self.mode == EncodingMode.QBA:
            word_ids, identity_ids, end_ids = self.encoder.encode(
                EncodingMode.QBA,
                question=item['question'],
                bridge=item['bridge'],
                answer=item['answer']
            )
        
        elif self.mode == EncodingMode.CONTINUOUS:
            word_ids, identity_ids, end_ids = self.encoder.encode(
                EncodingMode.CONTINUOUS,
                text=item['text'],
                chunk_size=self.chunk_size
            )
        
        # Truncate if needed
        if len(word_ids) > self.max_seq_len:
            word_ids = word_ids[:self.max_seq_len]
            identity_ids = identity_ids[:self.max_seq_len]
            end_ids = end_ids[:self.max_seq_len]
        
        # Create input/target pairs
        # Input: word_ids[:-1], Target: word_ids[1:]
        input_ids = torch.tensor(word_ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(word_ids[1:], dtype=torch.long)
        identity_ids = torch.tensor(identity_ids[:-1], dtype=torch.long)
        end_ids = torch.tensor(end_ids[1:], dtype=torch.long)
        
        return input_ids, target_ids, identity_ids, end_ids


def collate_mvec_batch(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for DataLoader.
    Pads sequences to the same length within a batch.
    
    Args:
        batch: List of (input_ids, target_ids, identity_ids, end_ids)
        
    Returns:
        Padded (input_ids, target_ids, identity_ids, end_ids) tensors
    """
    input_ids, target_ids, identity_ids, end_ids = zip(*batch)
    
    # Pad sequences
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=1)  # 1 = <pad>
    target_ids_padded = pad_sequence(target_ids, batch_first=True, padding_value=1)
    identity_ids_padded = pad_sequence(identity_ids, batch_first=True, padding_value=0)
    end_ids_padded = pad_sequence(end_ids, batch_first=True, padding_value=0)
    
    return input_ids_padded, target_ids_padded, identity_ids_padded, end_ids_padded


# ============================================================================
# Testing and validation
# ============================================================================

if __name__ == "__main__":
    import tempfile
    import os
    
    print("Testing MVecDataset with all modes")
    print("=" * 70)
    
    # Create test CSV files for each mode
    
    # Test 1: QA mode
    print("\nTest 1: QA Mode")
    print("-" * 70)
    
    qa_csv = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
    qa_csv.write("question,answer\n")
    qa_csv.write("What is 2+2?,Four\n")
    qa_csv.write("Who wrote Hamlet?,Shakespeare\n")
    qa_csv.write("What is the capital of France?,Paris\n")
    qa_csv.close()
    
    encoder = MVecEncoder()
    encoder.build_vocab([
        "What is 2+2?", "Four",
        "Who wrote Hamlet?", "Shakespeare",
        "What is the capital of France?", "Paris"
    ])
    
    qa_dataset = MVecDataset(qa_csv.name, encoder, max_seq_len=64, mode=EncodingMode.QA)
    print(f"QA Dataset size: {len(qa_dataset)}")
    
    # Get a sample
    input_ids, target_ids, identity_ids, end_ids = qa_dataset[0]
    print(f"Sample 0 shapes: input={input_ids.shape}, target={target_ids.shape}")
    print(f"Identity channels: {identity_ids.tolist()}")
    print(f"END markers: {end_ids.tolist()}")
    
    os.unlink(qa_csv.name)
    
    # Test 2: QBA mode
    print("\n\nTest 2: QBA Mode")
    print("-" * 70)
    
    qba_csv = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
    qba_csv.write("question,bridge,answer\n")
    qba_csv.write("What is 2+2?,math addition,Four\n")
    qba_csv.write("Who wrote Hamlet?,literature Shakespeare,Shakespeare wrote it\n")
    qba_csv.close()
    
    encoder2 = MVecEncoder()
    encoder2.build_vocab([
        "What is 2+2?", "math addition", "Four",
        "Who wrote Hamlet?", "literature Shakespeare", "Shakespeare wrote it"
    ])
    
    qba_dataset = MVecDataset(qba_csv.name, encoder2, max_seq_len=64, mode=EncodingMode.QBA)
    print(f"QBA Dataset size: {len(qba_dataset)}")
    
    input_ids, target_ids, identity_ids, end_ids = qba_dataset[0]
    print(f"Sample 0 shapes: input={input_ids.shape}, target={target_ids.shape}")
    print(f"Identity channels: {identity_ids.tolist()}")
    print(f"END markers: {end_ids.tolist()}")
    
    os.unlink(qba_csv.name)
    
    # Test 3: CONTINUOUS mode
    print("\n\nTest 3: CONTINUOUS Mode")
    print("-" * 70)
    
    cont_csv = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
    cont_csv.write("text\n")
    cont_csv.write("Once upon a time there was a wise old sage.\n")
    cont_csv.write("The sage lived in the mountains.\n")
    cont_csv.close()
    
    encoder3 = MVecEncoder()
    encoder3.build_vocab([
        "Once upon a time there was a wise old sage.",
        "The sage lived in the mountains."
    ])
    
    # Test without chunking
    cont_dataset = MVecDataset(
        cont_csv.name, 
        encoder3, 
        max_seq_len=64, 
        mode=EncodingMode.CONTINUOUS
    )
    print(f"CONTINUOUS Dataset size (no chunking): {len(cont_dataset)}")
    
    input_ids, target_ids, identity_ids, end_ids = cont_dataset[0]
    print(f"Sample 0 shapes: input={input_ids.shape}, target={target_ids.shape}")
    print(f"Identity channels (should all be 2=VISIBLE): {set(identity_ids.tolist())}")
    print(f"END markers: {end_ids.tolist()}")
    
    # Test with chunking
    cont_dataset_chunked = MVecDataset(
        cont_csv.name, 
        encoder3, 
        max_seq_len=64, 
        mode=EncodingMode.CONTINUOUS,
        chunk_size=5
    )
    print(f"\nCONTINUOUS Dataset size (chunk_size=5): {len(cont_dataset_chunked)}")
    
    input_ids, target_ids, identity_ids, end_ids = cont_dataset_chunked[0]
    print(f"Sample 0 shapes: input={input_ids.shape}, target={target_ids.shape}")
    print(f"END markers (with chunking): {end_ids.tolist()}")
    
    os.unlink(cont_csv.name)
    
    # Test 4: Batch collation
    print("\n\nTest 4: Batch Collation")
    print("-" * 70)
    
    from torch.utils.data import DataLoader
    
    loader = DataLoader(qa_dataset, batch_size=2, collate_fn=collate_mvec_batch)
    batch = next(iter(loader))
    
    print(f"Batch input shape: {batch[0].shape}")
    print(f"Batch target shape: {batch[1].shape}")
    print(f"Batch identity shape: {batch[2].shape}")
    print(f"Batch end shape: {batch[3].shape}")
    
    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
