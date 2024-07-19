"""
MIT License
Copyright (c) 2024 Yaochen Zhu
"""

import re
import os
import sys
import pickle
import fsspec
import argparse
from tqdm import tqdm

import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from accelerate import Accelerator
from scipy.sparse import load_npz
from transformers import GPT2Model, GPT2Config
from tokenizer import TokenizerWithUserItemIDTokensBatch
from data import CollaborativeGPTGeneratorBatch, UserItemContentGPTDatasetBatch
from model import GPT4RecommendationBaseModel, CollaborativeGPTwithItemLMHeadBatch, ContentGPTForUserItemWithLMHeadBatch

sys.path.append("libs")

def save_file(source_path, dest_path, mode_source, mode_dest):
    """Generalized function to save files from source to destination."""
    with fsspec.open(source_path, mode_source) as src_file:
        content = src_file.read()
    with fsspec.open(dest_path, mode_dest) as dest_file:
        dest_file.write(content)

def setup_directories():
    """Ensure that required directories exist."""
    if not os.path.exists(local_root):
        os.makedirs(local_root, exist_ok=True)

def load_config():
    """Load the model configuration."""
    return GPT2Config(**_config)

def load_pretrained_weights(model, path):
    """Load pretrained weights into the model."""
    model.load_state_dict(torch.load(path), strict=False)

def save_model_embeddings(model, save_path, lambda_V, is_main):
    """Save model embeddings if the current process is the main process."""
    if is_main:
        user_emb_path = os.path.join(save_path, f"user_embeddings_{lambda_V}.pt")
        item_emb_path = os.path.join(save_path, f"item_embeddings_{lambda_V}.pt")
        torch.save(model.base_model.user_embeddings.state_dict(), user_emb_path)
        torch.save(model.base_model.item_embeddings.state_dict(), item_emb_path)
        save_remote(user_emb_path, os.path.join(save_path, f"user_embeddings_{lambda_V}.pt"), "rb", "wb")
        save_remote(item_emb_path, os.path.join(save_path, f"item_embeddings_{lambda_V}.pt"), "rb", "wb")

def main():
    # Define the accelerator
    accelerator = Accelerator()
    device = accelerator.device

    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Specify the dataset for experiment")
    parser.add_argument("--lambda_V", type=str, help="Specify lambda_V for regularization")
    args = parser.parse_args()
    dataset = args.dataset
    lambda_V = float(args.lambda_V)
    
    accelerator.print("-----Current Setting-----")
    accelerator.print(f"dataset: {dataset}")
    accelerator.print(f"lambda_V: {lambda_V}")

    num_gpus = torch.cuda.device_count()
    accelerator.print(f"num_gpus: {num_gpus}")

    # Initialize paths and ensure directories exist
    setup_directories()
    data_root = os.path.join(gpt2_server_root, "dataset", dataset)
    meta_path = os.path.join(data_root, "meta.pkl")

    # Load dataset info
    accelerator.print("-----Begin Obtaining Dataset Info-----")
    with fsspec.open(meta_path, "rb") as f:
        meta_data = pickle.load(f)
    num_users, num_items = meta_data["num_users"], meta_data["num_items"]
    accelerator.print(f"num_users: {num_users}")
    accelerator.print(f"num_items: {num_items}")
    accelerator.print("-----End Obtaining Dataset Info-----\n")

    # Obtain tokenizer
    accelerator.print("-----Begin Obtaining the Tokenizer-----")
    tokenizer_root = os.path.join(gpt2_server_root, "model", "pretrained", "tokenizer")
    remote_vocab_file = os.path.join(tokenizer_root, "vocab_file.json")
    remote_merges_file = os.path.join(tokenizer_root, "merges.txt")
    vocab_file = os.path.join(local_root, "vocab_file.json")
    merges_file = os.path.join(local_root, "merges.txt")

    if accelerator.is_main_process:
        save_file(remote_vocab_file, vocab_file, "r", "w")
        save_file(remote_merges_file, merges_file, "r", "w")
    accelerator.wait_for_everyone()

    tokenizer = TokenizerWithUserItemIDTokensBatch(vocab_file, merges_file, num_users, num_items)
    accelerator.print("Success!")
    accelerator.print("-----End Obtaining the Tokenizer-----\n")

    # Load review data
    accelerator.print("-----Begin Obtaining the Review Data Generator-----")
    review_path = os.path.join(data_root, "user_item_texts", "review.pkl")
    review_data_gen = UserItemContentGPTDatasetBatch(tokenizer, review_path)
    accelerator.print("Success!")
    accelerator.print("-----End Obtaining the Review Data Generator-----\n")

    # Load collaborative data
    accelerator.print("-----Begin Obtaining the Collaborative Data Generator-----")
    remote_train_mat_path = os.path.join(data_root, "train_matrix.npz")
    local_train_mat_path = os.path.join(local_root, "train_matrix.npz")
    if accelerator.is_main_process:
        save_file(remote_train_mat_path, local_train_mat_path, "rb", "wb")
    accelerator.wait_for_everyone()
    
    train_mat = load_npz(local_train_mat_path)
    collaborative_data_gen = CollaborativeGPTGeneratorBatch(tokenizer, train_mat)
    accelerator.print("Success!")
    accelerator.print("-----End Obtaining the Collaborative Data Generator-----\n")

    # Setup model configuration
    accelerator.print("-----Begin Setting Up the Config-----")
    config = load_config()
    config.num_users = num_users
    config.num_items = num_items
    accelerator.print("Success!")
    accelerator.print("-----End Setting Up the Config-----\n")

    # Instantiate models
    accelerator.print("-----Begin Instantiating the Pretrained GPT Model-----")
    gpt2model = GPT2Model(config)
    pretrained_root = os.path.join(gpt2_server_root, "model", "pretrained")
    remote_pretrained_weights_path = os.path.join(pretrained_root, "gpt2", "pytorch_model.bin")
    local_pretrained_weights_path = os.path.join(local_root, "gpt2", "pytorch_model.bin")
    if accelerator.is_main_process:
        save_file(remote_pretrained_weights_path, local_pretrained_weights_path, "rb", "wb")
    accelerator.wait_for_everyone()
    load_pretrained_weights(gpt2model, local_pretrained_weights_path)
    accelerator.print("Success!")
    accelerator.print("-----End Instantiating the Pretrained GPT Model-----\n")

    # Create content model
    accelerator.print("-----Begin Instantiating the Content GPT Model-----")
    content_base_model = GPT4RecommendationBaseModel(config, gpt2model)
    content_model = ContentGPTForUserItemWithLMHeadBatch(config, content_base_model)
    accelerator.print("Success!")
    accelerator.print("-----End Instantiating the Content GPT Model-----\n")

    # Freeze non-trainable parameters
    for name, param in content_model.named_parameters():
        if 'user_embeddings' not in name and 'item_embeddings' not in name:
            param.requires_grad = False
    accelerator.print("-----Trainable Parameters-----")
    for name, param in content_model.named_parameters():
        if param.requires_grad:
            accelerator.print(f"{name} : {param.shape}")
    accelerator.print("\n-----Non-Trainable Parameters-----")
    for name, param in content_model.named_parameters():
        if not param.requires_grad:
            accelerator.print(f"{name} : {param.shape}")

    # Create collaborative model
    accelerator.print("-----Begin Instantiating the Collaborative GPT Model-----")
    collaborative_base_model = GPT4RecommendationBaseModel(config, gpt2model)
    collaborative_model = CollaborativeGPTwithItemLMHeadBatch(config, collaborative_base_model)
    accelerator.print("Success!")
    accelerator.print("-----End Instantiating the Collaborative GPT Model-----\n")

    # Freeze non-trainable parameters
    for name, param in collaborative_model.named_parameters():
        if 'user_embeddings' not in name and 'item_embeddings' not in name:
            param.requires_grad = False
    accelerator.print("-----Trainable Parameters-----")
    for name, param in collaborative_model.named_parameters():
        if param.requires_grad:
            accelerator.print(f"{name} : {param.shape}")
    accelerator.print("\n-----Non-Trainable Parameters-----")
    for name, param in collaborative_model.named_parameters():
        if not param.requires_grad:
            accelerator.print(f"{name} : {param.shape}")

    # Setup training details
    accelerator.print("-----Begin Setting Up the Training Details-----")
    learning_rate = 1e-3
    batch_size = 20
    num_pretrained_epochs = 10
    num_epochs = 100

    # Create data loaders
    accelerator.print("-----Begin Creating the DataLoader-----")
    review_data_loader = DataLoader(review_data_gen, batch_size=batch_size, collate_fn=review_data_gen.collate_fn)
    collaborative_data_loader = DataLoader(collaborative_data_gen, batch_size=batch_size, collate_fn=collaborative_data_gen.collate_fn)
    accelerator.print("Success!")
    accelerator.print("-----End Creating the DataLoader-----\n")

    # Prepare models and optimizers
    optimizer_content = optim.AdamW(content_model.parameters(), lr=learning_rate)
    optimizer_collaborative = optim.AdamW(collaborative_model.parameters(), lr=learning_rate)
    content_model, optimizer_content = accelerator.prepare(content_model, optimizer_content)
    collaborative_model, optimizer_collaborative = accelerator.prepare(collaborative_model, optimizer_collaborative)

    # Training loop for content model
    for epoch in range(num_pretrained_epochs):
        content_model.train()
        for batch in tqdm(review_data_loader, desc=f"Content Epoch {epoch}"):
            optimizer_content.zero_grad()
            input_ids, labels = batch
            outputs = content_model(input_ids, labels=labels)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer_content.step()
            accelerator.print(f"Content Epoch {epoch}, Loss: {loss.item()}")

    # Training loop for collaborative model
    for epoch in range(num_epochs):
        collaborative_model.train()
        for batch in tqdm(collaborative_data_loader, desc=f"Collaborative Epoch {epoch}"):
            optimizer_collaborative.zero_grad()
            input_ids, labels = batch
            outputs = collaborative_model(input_ids, labels=labels)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer_collaborative.step()
            accelerator.print(f"Collaborative Epoch {epoch}, Loss: {loss.item()}")

    # Save model embeddings
    accelerator.print("-----Begin Saving Model Embeddings-----")
    save_model_embeddings(content_model, local_root, lambda_V, accelerator.is_main_process)
    save_model_embeddings(collaborative_model, local_root, lambda_V, accelerator.is_main_process)
    accelerator.print("Success!")
    accelerator.print("-----End Saving Model Embeddings-----\n")

if __name__ == "__main__":
    main()

