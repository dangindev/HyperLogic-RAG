import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
# from torchvision import transforms
# Switch to manual transforms to avoid torchvision dependency issues
import numpy as np

# Use R2Gen-compatible tokenizer for consistent BLEU calculation
from src.utils.r2gen_tokenizer import R2GenTokenizer

# Shared tokenizer instance (created once per training run)
_SHARED_TOKENIZER = None

def get_shared_tokenizer(ann_path, max_length=128):
    """Get or create shared tokenizer instance."""
    global _SHARED_TOKENIZER
    if _SHARED_TOKENIZER is None:
        print(f"Initializing R2GenTokenizer from {ann_path}...")
        _SHARED_TOKENIZER = R2GenTokenizer(ann_path, threshold=3, max_length=max_length)
    return _SHARED_TOKENIZER


class MIMICCXRDataset(Dataset):
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        manifest_file = config['dataset'].get('manifest_file', 'mimic_cxr_clean.jsonl')
        self.data_path = os.path.join(config['dataset']['data_root'], manifest_file)
        self.max_length = config['dataset']['max_length']
        
        self.samples = []
        if os.path.exists(self.data_path):
            with open(self.data_path, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    # Support both 'val' and 'validate' split names
                    entry_split = entry['split'].strip()
                    if entry_split == split or (split == 'validate' and entry_split == 'val'):
                        self.samples.append(entry)
        else:
            print(f"Warning: Data file {self.data_path} not found. Dataset will be empty.")
        
        # Use R2Gen-compatible tokenizer (word-level, same as R2Gen for BLEU)
        self.tokenizer = get_shared_tokenizer(self.data_path, max_length=self.max_length)
        
        self.image_size = config['dataset']['image_size']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_paths = sample['image_path']
        if not isinstance(image_paths, list):
            image_paths = [image_paths]
            
        use_multi_image = self.config['dataset'].get('use_multi_image', False)
        # For multi-image: keep exactly 2 (pad or slice)
        if use_multi_image:
            if len(image_paths) == 0:
                image_paths = ["", ""]
            elif len(image_paths) == 1:
                image_paths = [image_paths[0], image_paths[0]]
            else:
                image_paths = image_paths[:2]
        else:
            image_paths = [image_paths[0]]
            
        report_text = sample['report']
        
        loaded_images = []
        
        for image_path in image_paths:
            # Path Correction for Flat Structure
            if not os.path.exists(image_path) and image_path != "":
                 try:
                    parts = image_path.split('/')
                    if len(parts) >= 4:
                        dicom_id = parts[-1]
                        study_id = parts[-2]
                        patient_id = parts[-3]
                        flat_filename = f"{patient_id}_{study_id}_{dicom_id}"
                        
                        # Robust candidate search
                        candidates = [
                            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(image_path)))), flat_filename),
                            os.path.join("data/mimic_cxr/images", flat_filename),
                            os.path.join("HyperLogicRAG/data/mimic_cxr/images", flat_filename),
                            flat_filename,
                        ]
                        
                        for c in candidates:
                            if os.path.exists(c):
                                image_path = c
                                break
                 except:
                     pass
    
            try:
                if image_path == "":
                    raise FileNotFoundError("Empty image path")
                image = Image.open(image_path).convert('RGB')
                # Manual Transform: Resize, ToTensor, Normalize
                image = image.resize((self.image_size, self.image_size))
                img_array = np.array(image).astype(np.float32) / 255.0
                # Normalize based on config
                norm_type = self.config['dataset'].get('normalization', 'imagenet')
                
                if norm_type == 'clip':
                    # BiomedCLIP / OpenCLIP stats
                    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
                    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
                else:
                    # ImageNet stats (default)
                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                    
                img_array = (img_array - mean) / std
                # HWC -> CHW
                img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1))
            except Exception as e:
                # print(f"Error loading image {image_path}: {e}") # Suppress to avoid spam
                img_tensor = torch.zeros((3, self.config['dataset']['image_size'], self.config['dataset']['image_size']))
            
            loaded_images.append(img_tensor)

        if use_multi_image:
            image = torch.stack(loaded_images) # [2, 3, H, W]
        else:
            image = loaded_images[0] # [3, H, W]

        # Use R2Gen tokenizer (returns dict with 'input_ids' and 'attention_mask' as lists)
        inputs = self.tokenizer.encode(report_text)
        
        return {
            'image': image,
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'image_path': image_path,
            'caption': report_text
        }

    def collate_fn(self, batch):
        """
        Custom collate function to pad sequences.
        """
        images = []
        input_ids = []
        attention_masks = []
        image_paths = []
        captions = []
        ids = []
        
        # HyperGraph specific
        hypergraph_indices = []
        context_ids = []
        context_masks = []
        
        for b in batch:
            images.append(b['image'])
            input_ids.append(b['input_ids'])
            attention_masks.append(b['attention_mask'])
            image_paths.append(b['image_path'])
            captions.append(b['caption'])
            ids.append(b.get('id', ''))
            
            if 'hypergraph_indices' in b:
                # Should be tensor? Or None?
                # If not present in all, handle it.
                pass 
                # Wait, MIMICCXR_RAG_Dataset doesn't seem to add hypergraph_indices in current view?
                # train_hyper_r2gen.py expects it? 
                # Let's check logic.
            
            if 'context_ids' in b:
                context_ids.append(b['context_ids'])
                context_masks.append(b['context_mask'])

        images = torch.stack(images)
        
        # Pad Input IDs
        # We need to pad to max length in batch or global max length?
        # R2Gen tokenizer encodes to fixed length (with padding)?
        # Let's check R2GenTokenizer.encode in r2gen_tokenizer.py
        # It has default padding=True, truncation=True, max_length=128.
        # So input_ids are ALREADY padded to max_length.
        # So we can just stack them!
        
        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)
        
        data = {
            'images': images,
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'image_paths': image_paths,
            'captions': captions,
            'ids': ids
        }
        
        if context_ids:
            # Context might also be padded by tokenizer?
            # Yes, datasets.py calls self.tokenizer.encode(retrieved_text)
            # So they are padded to max_length.
            data['context_ids'] = torch.stack(context_ids)
            data['context_mask'] = torch.stack(context_masks)
            
        return data


class MIMICCXR_RAG_Dataset(MIMICCXRDataset):
    def __init__(self, config, split='train'):
        super().__init__(config, split)
        self.rag_index = {}
        retrieval_file = config['dataset'].get('retrieval_file')
        if retrieval_file and os.path.exists(os.path.join(config['dataset']['data_root'], retrieval_file)):
            print(f"Loading retrieval index from {retrieval_file}...")
            with open(os.path.join(config['dataset']['data_root'], retrieval_file), 'r') as f:
                self.rag_index = json.load(f)
        else:
            print(f"Warning: Retrieval file {retrieval_file} not found. Context will be empty.")

    def __getitem__(self, idx):
        # Get standard sample
        data = super().__getitem__(idx)
        
        # Get Retrieval Context
        sample = self.samples[idx]
        image_path = sample['image_path'] # Used as key
        if isinstance(image_path, list):
            image_path_key = image_path[0]
        else:
            image_path_key = image_path
        
        retrieved_text = ""
        if image_path_key in self.rag_index:
            retrieved_text = self.rag_index[image_path_key]
        
        # Tokenize Context using R2Gen tokenizer
        context_inputs = self.tokenizer.encode(retrieved_text)
        
        data['context_ids'] = torch.tensor(context_inputs['input_ids'], dtype=torch.long)
        data['context_mask'] = torch.tensor(context_inputs['attention_mask'], dtype=torch.long)
        
        return data

def get_dataloader(config, split='train'):
    if config['model'].get('type') == 'rag':
        dataset = MIMICCXR_RAG_Dataset(config, split)
    else:
        dataset = MIMICCXRDataset(config, split)
        
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=config['dataset']['batch_size'], 
        shuffle=(split=='train'),
        num_workers=config['dataset']['num_workers']
    )

