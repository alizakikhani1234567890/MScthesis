"""
MIT License
Copyright (c) 2024 Yaochen Zhu
"""

import random
import fsspec
import pickle
import torch
from torch.utils.data import Dataset


class CollaborativeGPTGeneratorBatch(Dataset):
    """
    Dataset class for generating collaborative GPT input batches.

    Args:
        tokenizer (TokenizerWithUserItemIDTokensBatch): Custom tokenizer instance.
        train_mat (np.ndarray): Matrix of user-item interactions.
        max_length (int, optional): Maximum length of the encoded sequences. Defaults to 1024.
    """
    def __init__(self, tokenizer, train_mat, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.train_mat = train_mat
        self.max_length = max_length
        self.num_users, self.num_items = train_mat.shape
            
    def __len__(self):
        return self.num_users

    def __getitem__(self, idx):
        prompt = f"user_{idx} has interacted with"
        return prompt, self.train_mat.getrow(idx).nonzero()[1]

    def collate_fn(self, batch):
        """
        Custom collate function to encode and pad the batch of texts.

        Args:
            batch (List[Tuple[str, np.ndarray]]): List of tuples containing the prompt and item IDs.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing the encoded and padded prompt IDs,
                main IDs, and attention masks.
        """
        prompt_texts, item_ids = zip(*batch)
        encoded_prompt = self.tokenizer.encode_batch(prompt_texts)
        item_tokens = [" ".join(f"item_{item_id}" for item_id in ids) for ids in item_ids]
        encoded_main = self.tokenizer.encode_batch(item_tokens)

        prompt_ids = torch.tensor(encoded_prompt[0])
        main_ids = torch.tensor(encoded_main[0])
        attention_mask = torch.cat((torch.tensor(encoded_prompt[1]), torch.tensor(encoded_main[1])), dim=1)

        total_length = prompt_ids.size(1) + main_ids.size(1)
        if total_length > self.max_length:
            excess_length = total_length - self.max_length
            main_ids = main_ids[:, :-excess_length]
            attention_mask = attention_mask[:, :-excess_length]

        return prompt_ids, main_ids, attention_mask


class UserItemContentGPTDatasetBatch(Dataset):
    """
    Dataset class for generating user-item content GPT input batches.

    Args:
        tokenizer (TokenizerWithUserItemIDTokensBatch): Custom tokenizer instance.
        filepath (str): Path to the pickle file containing the descriptions.
        max_length (int, optional): Maximum length of the encoded sequences. Defaults to 1024.
    """
    def __init__(self, tokenizer, filepath, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        assert filepath.endswith(".pkl"), "Filepath must end with .pkl"

        with fsspec.open(filepath, 'rb') as file:
            self.data = pickle.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt_text, main_text = self.data[idx]
        return prompt_text, main_text

    def collate_fn(self, batch):
        """
        Custom collate function to encode and pad the batch of texts.

        Args:
            batch (List[Tuple[str, str]]): List of tuples containing the prompt and main texts.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing the encoded and padded prompt IDs,
                main IDs, and attention masks.
        """
        prompt_texts, main_texts = zip(*batch)
        encoded_prompt = self.tokenizer.encode_batch(prompt_texts)
        encoded_main = self.tokenizer.encode_batch(main_texts)

        prompt_ids = torch.tensor(encoded_prompt[0])
        main_ids = torch.tensor(encoded_main[0])
        attention_mask = torch.cat((torch.tensor(encoded_prompt[1]), torch.tensor(encoded_main[1])), dim=1)

        total_length = prompt_ids.size(1) + main_ids.size(1)
        if total_length > self.max_length:
            excess_length = total_length - self.max_length
            main_ids = main_ids[:, :-excess_length]
            attention_mask = attention_mask[:, :-excess_length]

        return prompt_ids, main_ids, attention_mask
    

class RecommendationGPTTrainGeneratorBatch(Dataset):
    """
    Dataset class for generating recommendation GPT input batches.

    Args:
        tokenizer (TokenizerWithUserItemIDTokensBatch): Custom tokenizer instance.
        train_mat (np.ndarray): Matrix of user-item interactions.
        max_length (int, optional): Maximum length of the encoded sequences. Defaults to 1024.
        predict_ratio (float, optional): The percentage of items to predict for each user (default: 0.2).
        shuffle (bool, optional): Whether to shuffle the input interactions (default: True).
    """
    def __init__(self, tokenizer, train_mat, max_length=1024, predict_ratio=0.2, shuffle=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.train_mat = train_mat
        self.max_length = max_length
        self.num_users, self.num_items = train_mat.shape
        self.predict_ratio = predict_ratio
        self.shuffle = shuffle

    def __len__(self):
        return self.num_users
    
    def __getitem__(self, idx):
        past_interactions = self.train_mat.getrow(idx).nonzero()[1]
        num_items_to_mask = max(1, int(len(past_interactions) * self.predict_ratio))
        masked_items = random.sample(past_interactions.tolist(), num_items_to_mask)
        input_interactions = [item if item not in masked_items else None for item in past_interactions]
        if self.shuffle:
            random.shuffle(input_interactions)

        input_prompt = f"user_{idx} has interacted with {' '.join(f'item_{item_id}' for item_id in input_interactions if item_id is not None)}"
        input_prompt += f", user_{idx} will interact with"

        target_matrix = torch.zeros(self.num_items, dtype=torch.float32)
        target_matrix[past_interactions] = 1.0
        item_ids = target_matrix.nonzero(as_tuple=True)[0]
        
        return input_prompt, target_matrix, item_ids

    def collate_fn(self, batch):
        """
        Custom collate function to encode and pad the batch of texts.

        Args:
            batch (List[Tuple[str, torch.Tensor]]): List of tuples containing the prompt and target matrix.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing the encoded and padded prompt IDs,
                target matrix, and attention mask.
        """
        prompt_texts, target_matrices, item_ids = zip(*batch)
        encoded_prompt = self.tokenizer.encode_batch(prompt_texts)
        target_matrices = torch.stack(target_matrices)
        item_tokens = [" ".join(f"item_{item_id}" for item_id in ids) for ids in item_ids]
        encoded_main = self.tokenizer.encode_batch(item_tokens)

        prompt_ids = torch.tensor(encoded_prompt[0])
        main_ids = torch.tensor(encoded_main[0])
        attention_mask = torch.tensor(encoded_prompt[1])

        total_length = prompt_ids.size(1)
        if total_length > self.max_length:
            excess_length = total_length - self.max_length
            prompt_ids = prompt_ids[:, :-excess_length]
            attention_mask = attention_mask[:, :-excess_length]

        return prompt_ids, target_matrices, attention_mask, main_ids


class RecommendationGPTTestGeneratorBatch(Dataset):
    """
    Dataset class for generating recommendation GPT input batches.

    Args:
        tokenizer (TokenizerWithUserItemIDTokensBatch): Custom tokenizer instance.
        train_mat (np.ndarray): Matrix of user-item interactions.
        test_mat (np.ndarray): Matrix of user-item test interactions.
        max_length (int, optional): Maximum length of the encoded sequences. Defaults to 1024.
        predict_ratio (float, optional): The percentage of items to predict for each user (default: 0.2).
        shuffle (bool, optional): Whether to shuffle the input interactions (default: True).
    """
    def __init__(self, tokenizer, train_mat, test_mat, max_length=1024, predict_ratio=0.2, shuffle=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.train_mat = train_mat
        self.test_mat = test_mat
        self.max_length = max_length
        self.num_users, self.num_items = train_mat.shape
        self.predict_ratio = predict_ratio
        self.shuffle = shuffle

    def __len__(self):
        return self.num_users
    
    def __getitem__(self, idx):
        input_interactions = self.train_mat.getrow(idx).nonzero()[1]
        if self.shuffle:
            random.shuffle(input_interactions)
        
        input_prompt = f"user_{idx} has interacted with {' '.join(f'item_{item_id}' for item_id in input_interactions)}"
        input_prompt += f", user_{idx} will interact with"
        
        train_interactions = self.train_mat.getrow(idx).nonzero()[1]
        train_matrix = torch.zeros(self.num_items, dtype=torch.float32)
        train_matrix[train_interactions] = 1.0
        
        target_interactions = self.test_mat.getrow(idx).nonzero()[1]
        target_matrix = torch.zeros(self.num_items, dtype=torch.float32)
        target_matrix[target_interactions] = 1.0
        
        return input_prompt, train_matrix, target_matrix

    def collate_fn(self, batch):
        """
        Custom collate function to encode and pad the batch of texts.

        Args:
            batch (List[Tuple[str, torch.Tensor]]): List of tuples containing the prompt and target matrix.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing the encoded and padded prompt IDs,
                train matrices, target matrices, and attention mask.
        """
        prompt_texts, train_matrices, target_matrices = zip(*batch)
        encoded_prompt = self.tokenizer.encode_batch(prompt_texts)
        train_matrices = torch.stack(train_matrices)
        target_matrices = torch.stack(target_matrices)

        prompt_ids = torch.tensor(encoded_prompt[0])
        attention_mask = torch.tensor(encoded_prompt[1])

        total_length = prompt_ids.size(1)
        if total_length > self.max_length:
            excess_length = total_length - self.max_length
            prompt_ids = prompt_ids[:, :-excess_length]
            attention_mask = attention_mask[:, :-excess_length]

        return prompt_ids, train_matrices, target_matrices, attention_mask

