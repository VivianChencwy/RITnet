#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Script for Eye State Classification

This script trains an eye state classifier by loading data from merged_data.csv
and extracting frames from the corresponding video file.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import cv2
import pandas as pd
from PIL import Image
from torchvision import transforms
import random

from eye_state_model import EyeStateClassifier, EyeStateClassifierLite, get_nparams


# Default transform: convert to tensor and normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


class VideoCSVDataset(Dataset):
    """
    Dataset that loads frames from video based on CSV timestamps and eye_state labels.
    
    Args:
        csv_path: Path to merged_data.csv containing eye_state labels
        video_path: Path to video file
        split: One of 'train', 'validation' (80/20 split)
        resize: Tuple of (width, height) to resize frames
        augment: Whether to apply data augmentation
        sample_rate: Sample every N-th frame to reduce dataset size
    """
    
    def __init__(self, csv_path, video_path, split='train', resize=(320, 240),
                 augment=True, sample_rate=10):
        self.resize = resize
        self.augment = augment and (split == 'train')
        self.transform = transform
        
        # CLAHE for preprocessing
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        
        # Load CSV
        print(f"Loading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Filter rows with valid eye_state (0 or 1)
        df = df[df['eye_state'].isin([0, 1])].reset_index(drop=True)
        print(f"  Total rows with valid eye_state: {len(df)}")
        
        # Sample every N-th row
        if sample_rate > 1:
            df = df.iloc[::sample_rate].reset_index(drop=True)
            print(f"  After sampling (rate={sample_rate}): {len(df)}")
        
        # Open video to get properties
        self.video_path = video_path
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        print(f"  Video FPS: {self.fps}, Total frames: {self.total_frames}")
        
        # Calculate frame indices from timestamps
        # Assuming lsl_timestamp is relative or we use row index
        df['frame_idx'] = df.index * sample_rate
        df = df[df['frame_idx'] < self.total_frames].reset_index(drop=True)
        
        # Train/validation split (80/20)
        total_samples = len(df)
        indices = list(range(total_samples))
        random.seed(42)
        random.shuffle(indices)
        
        split_idx = int(0.8 * total_samples)
        if split == 'train':
            indices = indices[:split_idx]
        else:
            indices = indices[split_idx:]
        
        self.df = df.iloc[indices].reset_index(drop=True)
        self.labels = self.df['eye_state'].tolist()
        self.frame_indices = self.df['frame_idx'].tolist()
        
        # Print dataset info
        open_count = self.labels.count(0)
        close_count = self.labels.count(1)
        print(f"  {split} set: {len(self.labels)} samples (open={open_count}, close={close_count})")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        frame_idx = self.frame_indices[idx]
        label = self.labels[idx]
        
        # Read frame from video
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            # Return a blank frame if read fails
            frame = np.zeros((self.resize[1], self.resize[0]), dtype=np.uint8)
        else:
            # Convert to grayscale
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Resize
            frame = cv2.resize(frame, self.resize)
        
        # Gamma correction
        table = 255.0 * (np.linspace(0, 1, 256) ** 0.8)
        frame = cv2.LUT(frame, table)
        
        # Data augmentation
        if self.augment:
            # Random horizontal flip
            if random.random() < 0.5:
                frame = cv2.flip(frame, 1)
            
            # Random brightness
            if random.random() < 0.2:
                factor = random.uniform(0.8, 1.2)
                frame = np.clip(frame * factor, 0, 255).astype(np.uint8)
        
        # CLAHE
        frame = self.clahe.apply(np.uint8(frame))
        
        # Convert to PIL and apply transform
        img = Image.fromarray(frame)
        img = self.transform(img)
        
        return img, label, frame_idx
    
    def get_class_weights(self):
        """Calculate class weights for handling class imbalance."""
        labels = np.array(self.labels)
        class_counts = np.bincount(labels)
        total = len(labels)
        weights = total / (len(class_counts) * class_counts)
        return torch.FloatTensor(weights)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Eye State Classifier from CSV + Video')
    
    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to data directory containing merged_data.csv')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to video file (optional, auto-detect from data_dir)')
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to CSV file (optional, defaults to merged_data.csv)')
    
    # Output
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output directory for trained models')
    
    # Model
    parser.add_argument('--model', type=str, default='densenet', choices=['densenet', 'lite'],
                        help='Model type: densenet or lite')
    parser.add_argument('--resize', type=str, default='320x240',
                        help='Resize frames to WxH')
    
    # Training
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--bs', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--sample_rate', type=int, default=10,
                        help='Sample every N-th frame from video')
    
    # GPU
    parser.add_argument('--useGPU', type=str, default='True',
                        help='Use GPU if available')
    
    args = parser.parse_args()
    return args


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on given dataloader."""
    model.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images, labels, _ = batch
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += loss.item()
            num_batches += 1
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    avg_loss = total_loss / max(num_batches, 1)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        images, labels, _ = batch
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})
    
    return total_loss / len(dataloader), correct / total


def plot_training_history(history, save_path):
    """Plot training history."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train')
    axes[0, 1].plot(history['val_acc'], label='Validation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # F1 Score
    axes[1, 0].plot(history['val_f1'], label='Validation F1')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning Rate
    axes[1, 1].plot(history['lr'])
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training history saved to {save_path}")


def find_video_in_dir(data_dir):
    """Find a video file in the data directory."""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    for f in os.listdir(data_dir):
        if any(f.lower().endswith(ext) for ext in video_extensions):
            return os.path.join(data_dir, f)
    return None


def main():
    args = parse_args()
    
    # Parse resize
    w, h = map(int, args.resize.split('x'))
    resize = (w, h)
    
    # Find CSV and video paths
    csv_path = args.csv if args.csv else os.path.join(args.data_dir, 'merged_data.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    video_path = args.video if args.video else find_video_in_dir(args.data_dir)
    if video_path is None or not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found in {args.data_dir}")
    
    print(f"CSV: {csv_path}")
    print(f"Video: {video_path}")
    
    # Device setup
    use_gpu = args.useGPU.lower() == 'true' and torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')
    print(f"Using device: {device}")
    
    if use_gpu:
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)
    
    # Create output directory
    log_dir = os.path.join(args.output_dir, 'logs', 'eye_state')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'models'), exist_ok=True)
    
    # Load datasets
    print("\n=== Loading Datasets ===")
    train_dataset = VideoCSVDataset(
        csv_path, video_path, split='train', resize=resize,
        augment=True, sample_rate=args.sample_rate
    )
    val_dataset = VideoCSVDataset(
        csv_path, video_path, split='validation', resize=resize,
        augment=False, sample_rate=args.sample_rate
    )
    
    # DataLoaders (num_workers=0 for Windows compatibility)
    num_workers = 0
    train_loader = DataLoader(
        train_dataset, batch_size=args.bs, shuffle=True,
        num_workers=num_workers, pin_memory=use_gpu
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.bs, shuffle=False,
        num_workers=num_workers, pin_memory=use_gpu
    )
    
    # Create model
    print("\n=== Creating Model ===")
    if args.model == 'densenet':
        model = EyeStateClassifier(in_channels=1, num_classes=2, dropout=True, prob=0.2)
    else:
        model = EyeStateClassifierLite(in_channels=1, num_classes=2)
    
    model = model.to(device)
    print(f"Model: {args.model}")
    print(f"Parameters: {get_nparams(model):,}")
    
    # Loss function with class weights
    class_weights = train_dataset.get_class_weights().to(device)
    print(f"Class weights: {class_weights.cpu().numpy()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
        'lr': []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    
    # Training loop
    print("\n=== Starting Training ===")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['lr'].append(current_lr)
        
        # Print metrics
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"  Val F1: {val_metrics['f1']:.4f}, LR: {current_lr:.6f}")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_classifier.pkl'))
            print(f"  ** Best model saved (Acc: {best_val_acc:.4f}) **")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(log_dir, 'final_classifier.pkl'))
    
    # Plot training history
    plot_training_history(history, os.path.join(log_dir, 'training_history.png'))
    
    # Print summary
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Best Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
    print(f"Best model saved to: {os.path.join(log_dir, 'best_classifier.pkl')}")
    print(f"Final model saved to: {os.path.join(log_dir, 'final_classifier.pkl')}")


if __name__ == '__main__':
    main()
