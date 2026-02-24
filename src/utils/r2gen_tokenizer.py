"""
R2Gen-compatible Tokenizer for HyperLogic RAG
Uses word-level tokenization matching R2Gen's approach for consistent BLEU calculation.
"""
import re
import json
from collections import Counter


class R2GenTokenizer:
    """
    Word-level tokenizer matching R2Gen's approach.
    
    Key features:
    - Clean report text (remove punctuation, lowercase)
    - Word-level tokens (not subword like BERT)
    - Compatible with R2Gen metrics calculation
    """
    
    def __init__(self, ann_path, threshold=3, max_length=128):
        """
        Args:
            ann_path: Path to annotation file (JSONL format)
            threshold: Minimum word frequency to include in vocab
            max_length: Maximum sequence length
        """
        self.ann_path = ann_path
        self.threshold = threshold
        self.max_length = max_length
        
        # Special tokens
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'
        
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        
        # Load annotations and build vocabulary
        self.ann = self._load_annotations()
        self.token2idx, self.idx2token = self._create_vocabulary()
        self.vocab_size = len(self.token2idx)
        
        print(f"✓ R2GenTokenizer initialized: vocab_size={self.vocab_size}")
    
    def _load_annotations(self):
        """Load annotations from JSONL file."""
        samples = []
        if self.ann_path.endswith('.jsonl'):
            with open(self.ann_path, 'r') as f:
                for line in f:
                    samples.append(json.loads(line))
        else:
            with open(self.ann_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    samples = data
                else:
                    # Assume dict with train/val/test splits
                    for split in ['train', 'val', 'test']:
                        if split in data:
                            samples.extend(data[split])
        return samples
    
    def _create_vocabulary(self):
        """Create vocabulary from training data."""
        total_tokens = []
        
        for example in self.ann:
            if example.get('split', 'train') == 'train':
                tokens = self.clean_report(example['report']).split()
                total_tokens.extend(tokens)
        
        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold]
        vocab.sort()
        
        # Build token <-> idx mappings
        token2idx = {
            self.pad_token: self.pad_token_id,
            self.unk_token: self.unk_token_id,
            self.bos_token: self.bos_token_id,
            self.eos_token: self.eos_token_id,
        }
        idx2token = {
            self.pad_token_id: self.pad_token,
            self.unk_token_id: self.unk_token,
            self.bos_token_id: self.bos_token,
            self.eos_token_id: self.eos_token,
        }
        
        for idx, token in enumerate(vocab, start=4):  # Start after special tokens
            token2idx[token] = idx
            idx2token[idx] = token
        
        return token2idx, idx2token
    
    def clean_report(self, report):
        """
        Clean report text following R2Gen's approach.
        - Lowercase
        - Remove punctuation (except periods for sentence boundaries)
        - Normalize whitespace
        """
        # R2Gen cleaning pipeline for MIMIC-CXR
        report = report.replace('\n', ' ').replace('__', '_')
        report = re.sub(r'\s+', ' ', report)  # Normalize whitespace
        report = report.replace('..', '.').strip().lower()
        
        # Split into sentences and clean each
        sentences = report.split('. ')
        cleaned_sentences = []
        for sent in sentences:
            # Remove punctuation except for sentence boundaries
            sent = re.sub(r'[,?;*!%^&_+():\-\[\]{}"\'/\\]', '', sent)
            sent = sent.strip()
            if sent:
                cleaned_sentences.append(sent)
        
        # Join with ' . ' to mark sentence boundaries (R2Gen style)
        cleaned = ' . '.join(cleaned_sentences)
        if cleaned and not cleaned.endswith('.'):
            cleaned += ' .'
        
        return cleaned
    
    def encode(self, text, add_special_tokens=True, padding=True, truncation=True):
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Add BOS/EOS tokens
            padding: Pad to max_length
            truncation: Truncate to max_length
        
        Returns:
            dict with 'input_ids' and 'attention_mask'
        """
        cleaned = self.clean_report(text)
        tokens = cleaned.split()
        
        # Convert to IDs
        ids = []
        if add_special_tokens:
            ids.append(self.bos_token_id)
        
        for token in tokens:
            if token in self.token2idx:
                ids.append(self.token2idx[token])
            else:
                ids.append(self.unk_token_id)
        
        if add_special_tokens:
            ids.append(self.eos_token_id)
        
        # Truncation
        if truncation and len(ids) > self.max_length:
            ids = ids[:self.max_length]
        
        # Attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(ids)
        
        # Padding
        if padding:
            pad_length = self.max_length - len(ids)
            if pad_length > 0:
                ids = ids + [self.pad_token_id] * pad_length
                attention_mask = attention_mask + [0] * pad_length
        
        return {
            'input_ids': ids,
            'attention_mask': attention_mask
        }
    
    def decode(self, ids, skip_special_tokens=True):
        """
        Decode token IDs to text.
        
        Args:
            ids: Token IDs (list or tensor)
            skip_special_tokens: Skip BOS/EOS/PAD tokens
        
        Returns:
            Decoded text string
        """
        if hasattr(ids, 'tolist'):
            ids = ids.tolist()
        
        tokens = []
        for idx in ids:
            if skip_special_tokens and idx in [self.pad_token_id, self.bos_token_id, self.eos_token_id]:
                continue
            if idx == self.eos_token_id:
                break  # Stop at EOS
            if idx in self.idx2token:
                tokens.append(self.idx2token[idx])
            else:
                tokens.append(self.unk_token)
        
        return ' '.join(tokens)
    
    def batch_decode(self, ids_batch, skip_special_tokens=True):
        """Decode a batch of token IDs."""
        return [self.decode(ids, skip_special_tokens) for ids in ids_batch]
    
    def __call__(self, text, **kwargs):
        """Convenience method to encode text."""
        return self.encode(text, **kwargs)
    
    def get_vocab_size(self):
        """Return vocabulary size."""
        return self.vocab_size


# Factory function for easy creation
def create_tokenizer(ann_path, threshold=3, max_length=128):
    """Create R2Gen-compatible tokenizer."""
    return R2GenTokenizer(ann_path, threshold=threshold, max_length=max_length)


if __name__ == "__main__":
    # Test tokenizer
    import os
    
    ann_path = "data/mimic_cxr_clean.jsonl"
    if os.path.exists(ann_path):
        tokenizer = R2GenTokenizer(ann_path, threshold=3, max_length=128)
        
        # Test encoding
        test_report = "There is no evidence of pneumothorax. The cardiac silhouette is within normal limits."
        encoded = tokenizer.encode(test_report)
        print(f"\nTest report: {test_report}")
        print(f"Encoded IDs: {encoded['input_ids'][:15]}...")
        print(f"Attention mask: {encoded['attention_mask'][:15]}...")
        
        # Test decoding
        decoded = tokenizer.decode(encoded['input_ids'])
        print(f"Decoded: {decoded}")
    else:
        print(f"Annotation file not found: {ann_path}")
