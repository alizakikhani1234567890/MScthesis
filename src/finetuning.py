import re
import os
import sys
import pickle
import fsspec
import argparse
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2Model, GPT2Config
from accelerate import Accelerator
from scipy.sparse import load_npz

sys.path.append("libs")
from tokenizer import TokenizerWithUserItemIDTokensBatch
from data import UserItemContentGPTDatasetBatch, RecommendationGPTTrainGeneratorBatch, RecommendationGPTTestGeneratorBatch
from model import GPT4RecommendationBaseModel, ContentGPTForUserItemWithLMHeadBatch, CollaborativeGPTwithItemRecommendHead
from util import Recall_at_k, NDCG_at_k

def save_file(source_path, dest_path, source_mode, dest_mode):
    with fsspec.open(source_path, source_mode) as src_file, fsspec.open(dest_path, dest_mode) as dest_file:
        dest_file.write(src_file.read())

def create_directory(path):
    os.makedirs(path, exist_ok=True)

def get_dataset_info(data_root, accelerator):
    meta_path = os.path.join(data_root, "meta.pkl")
    with fsspec.open(meta_path, "rb") as f:
        meta_data = pickle.load(f)
    num_users = meta_data["num_users"]
    num_items = meta_data["num_items"]
    accelerator.print(f"num_users: {num_users}, num_items: {num_items}")
    return num_users, num_items

def get_tokenizer(server_root, local_root, num_users, num_items, accelerator):
    tokenizer_root = os.path.join(server_root, "model", "pretrained", "tokenizer")
    remote_vocab_file = os.path.join(tokenizer_root, "vocab_file.json")
    remote_merges_file = os.path.join(tokenizer_root, "merges.txt")
    vocab_file = os.path.join(local_root, "vocab_file.json")
    merges_file = os.path.join(local_root, "merges.txt")

    if accelerator.is_main_process:
        save_file(remote_vocab_file, vocab_file, "r", "w")
        save_file(remote_merges_file, merges_file, "r", "w")
    accelerator.wait_for_everyone()

    tokenizer = TokenizerWithUserItemIDTokensBatch(vocab_file, merges_file, num_users, num_items)
    accelerator.print("Tokenizer loaded successfully!")
    return tokenizer

def get_data_generators(data_root, local_root, tokenizer, accelerator):
    review_path = os.path.join(data_root, "user_item_texts", "review.pkl")
    review_data_gen = UserItemContentGPTDatasetBatch(tokenizer, review_path)

    remote_train_mat_path = os.path.join(data_root, "train_matrix.npz")
    local_train_mat_path = os.path.join(local_root, "train_matrix.npz")
    if accelerator.is_main_process:
        save_file(remote_train_mat_path, local_train_mat_path, "rb", "wb")

    remote_val_mat_path = os.path.join(data_root, "val_matrix.npz")
    local_val_mat_path = os.path.join(local_root, "val_matrix.npz")
    if accelerator.is_main_process:
        save_file(remote_val_mat_path, local_val_mat_path, "rb", "wb")
    accelerator.wait_for_everyone()

    train_mat = load_npz(local_train_mat_path)
    train_data_gen = RecommendationGPTTrainGeneratorBatch(tokenizer, train_mat)

    val_mat = load_npz(local_val_mat_path)
    val_data_gen = RecommendationGPTTestGeneratorBatch(tokenizer, train_mat, val_mat)

    accelerator.print("Data generators loaded successfully!")
    return review_data_gen, train_data_gen, val_data_gen

def load_pretrained_weights(model, remote_weights_path, local_weights_path, accelerator, device):
    if accelerator.is_main_process:
        save_file(remote_weights_path, local_weights_path, "rb", "wb")
    accelerator.wait_for_everyone()
    model.load_state_dict(torch.load(local_weights_path, map_location=device), strict=False)

def get_model_config(config_params):
    return GPT2Config(**config_params)

def instantiate_model(config, pretrained_weights_path, local_weights_path, accelerator, device):
    gpt2model = GPT2Model(config)
    load_pretrained_weights(gpt2model, pretrained_weights_path, local_weights_path, accelerator, device)
    return gpt2model

def initialize_pretrained_embeddings(model, user_emb_path, item_emb_path, accelerator, device):
    if accelerator.is_main_process:
        save_file(user_emb_path, local_user_emb_path, "rb", "wb")
        save_file(item_emb_path, local_item_emb_path, "rb", "wb")
    accelerator.wait_for_everyone()

    model.user_embeddings.load_state_dict(torch.load(local_user_emb_path, map_location=device))
    model.item_embeddings.load_state_dict(torch.load(local_item_emb_path, map_location=device))

def freeze_non_trainable_params(model):
    for name, param in model.named_parameters():
        if 'user_embeddings' not in name and 'item_embeddings' not in name:
            param.requires_grad = False

def setup_dataloaders(train_data_gen, val_data_gen, review_data_gen, batch_size, val_batch_size):
    train_data_loader = DataLoader(train_data_gen, batch_size=batch_size, collate_fn=train_data_gen.collate_fn)
    val_data_loader = DataLoader(val_data_gen, batch_size=val_batch_size, collate_fn=val_data_gen.collate_fn)
    review_data_loader = DataLoader(review_data_gen, batch_size=batch_size, collate_fn=review_data_gen.collate_fn)
    return train_data_loader, val_data_loader, review_data_loader

def main():
    # Define the accelerator
    accelerator = Accelerator()
    device = accelerator.device
    
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="specify the dataset for experiment")
    parser.add_argument("--lambda_V", type=str, help="specify the dataset for experiment")
    args = parser.parse_args()
    
    dataset = args.dataset
    lambda_V = float(args.lambda_V)
    
    accelerator.print("-----Current Setting-----")
    accelerator.print(f"dataset: {dataset}, lambda_V: {args.lambda_V}")
    
    # Define the number of GPUs to be used
    num_gpus = torch.cuda.device_count()
    accelerator.print(f"num_gpus: {num_gpus}")
    
    server_root = "hdfs://llm4rec"
    local_root = "tmp"
    create_directory(local_root)

    # Get dataset info
    data_root = os.path.join(server_root, "dataset", dataset)
    num_users, num_items = get_dataset_info(data_root, accelerator)

    # Obtain the tokenizer
    tokenizer = get_tokenizer(server_root, local_root, num_users, num_items, accelerator)
    
    # Define the data generators
    review_data_gen, train_data_gen, val_data_gen = get_data_generators(data_root, local_root, tokenizer, accelerator)

    # Set up the model config
    config_params = {
        "activation_function": "gelu_new",
        "architectures": ["GPT2LMHeadModel"],
        "attn_pdrop": 0.1,
        "bos_token_id": 50256,
        "embd_pdrop": 0.1,
        "eos_token_id": 50256,
        "initializer_range": 0.02,
        "layer_norm_epsilon": 1e-05,
        "model_type": "gpt2",
        "n_ctx": 1024,
        "n_embd": 768,
        "n_head": 12,
        "n_layer": 12,
        "n_positions": 1024,
        "resid_pdrop": 0.1,
        "summary_activation": None,
        "summary_first_dropout": 0.1,
        "summary_proj_to_labels": True,
        "summary_type": "cls_index",
        "summary_use_proj": True,
        "task_specific_params": {
            "text-generation": {"do_sample": True, "max_length": 50}
        },
        "vocab_size": 50257
    }
    config = get_model_config(config_params)
    config.num_users = num_users
    config.num_items = num_items

    # Instantiate the pretrained GPT2 model
    pretrained_root = os.path.join(server_root, "model", "pretrained")
    remote_pretrained_weights_path = os.path.join(pretrained_root, "gpt2", "pytorch_model.bin")
    local_pretrained_weights_path = os.path.join(local_root, "gpt2", "pytorch_model.bin")
    gpt2model = instantiate_model(config, remote_pretrained_weights_path, local_pretrained_weights_path, accelerator, device)
    
    # Instantiate the content GPT model
    content_base_model = GPT4RecommendationBaseModel(config, gpt2model)
    pretrained_root = os.path.join(server_root, "model", dataset, "content")
    remote_pretrained_user_emb_path = os.path.join(pretrained_root, f"user_embeddings_{args.lambda_V}.pt") 
    remote_pretrained_item_emb_path = os.path.join(pretrained_root, f"item_embeddings_{args.lambda_V}.pt") 
    local_pretrained_user_emb_path = os.path.join(local_root, "user_embeddings.pt")
    local_pretrained_item_emb_path = os.path.join(local_root, "item_embeddings.pt")
    initialize_pretrained_embeddings(content_base_model, remote_pretrained_user_emb_path, remote_pretrained_item_emb_path, accelerator, device)
    
    content_gpt_model = ContentGPTForUserItemWithLMHeadBatch(config, gpt2model, content_base_model)
    if accelerator.is_main_process:
        save_file(os.path.join(pretrained_root, "pytorch_model.bin"), os.path.join(local_root, "pytorch_model.bin"), "rb", "wb")
    accelerator.wait_for_everyone()
    content_gpt_model.load_state_dict(torch.load(os.path.join(local_root, "pytorch_model.bin"), map_location=device), strict=False)

    # Instantiate the collaborative filtering model
    collab_gpt_model = CollaborativeGPTwithItemRecommendHead(config, content_gpt_model.transformer, content_base_model.user_embeddings)
    pretrained_root = os.path.join(server_root, "model", dataset, "pretrained")
    remote_pretrained_cf_model_path = os.path.join(pretrained_root, "pytorch_model.bin")
    local_pretrained_cf_model_path = os.path.join(local_root, "pytorch_model.bin")
    load_pretrained_weights(collab_gpt_model, remote_pretrained_cf_model_path, local_pretrained_cf_model_path, accelerator, device)
    
    freeze_non_trainable_params(collab_gpt_model)

    # Setup data loaders
    batch_size = 32
    val_batch_size = 256
    train_data_loader, val_data_loader, review_data_loader = setup_dataloaders(train_data_gen, val_data_gen, review_data_gen, batch_size, val_batch_size)
    
    # Training and Evaluation
    num_epochs = 10
    optimizer = optim.Adam(collab_gpt_model.parameters(), lr=5e-5)
    
    collab_gpt_model, optimizer, train_data_loader, val_data_loader = accelerator.prepare(
        collab_gpt_model, optimizer, train_data_loader, val_data_loader
    )

    accelerator.print("Training model...")
    best_recall = -1.0
    collab_gpt_model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in tqdm(train_data_loader):
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = collab_gpt_model(**batch).loss
            accelerator.backward(loss)
            optimizer.step()
            total_loss += loss.item()
        
        average_loss = total_loss / len(train_data_loader)
        accelerator.print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}")

        # Evaluation
        collab_gpt_model.eval()
        recall_at_k = Recall_at_k(10)
        ndcg_at_k = NDCG_at_k(10)

        with torch.no_grad():
            for batch in val_data_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                item_indices = batch["item_indices"]
                relevance_scores = collab_gpt_model(**batch).logits
                recall_at_k.update(relevance_scores, item_indices)
                ndcg_at_k.update(relevance_scores, item_indices)
        
        recall_result = recall_at_k.compute()
        ndcg_result = ndcg_at_k.compute()
        accelerator.print(f"Recall@10: {recall_result:.4f}, NDCG@10: {ndcg_result:.4f}")

        if recall_result > best_recall:
            best_recall = recall_result
            accelerator.print(f"Saving best model with Recall@10: {best_recall:.4f}")
            accelerator.save_state(os.path.join(local_root, "best_model.pt"))

if __name__ == "__main__":
    main()

