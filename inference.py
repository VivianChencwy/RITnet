#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference Script for Eye State Classification

This script uses a trained model to predict eye states (open/close) for video files
and generates CSV annotation files and annotated videos.
"""

import os
import cv2
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from eye_state_model import EyeStateClassifier, EyeStateClassifierLite


def parse_args():
    parser = argparse.ArgumentParser(description='Inference for Eye State Classification')
    
    # Input
    parser.add_argument('--video', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model weights (.pkl file)')
    
    # Model
    parser.add_argument('--model_type', type=str, default='densenet', choices=['densenet', 'lite'],
                        help='Model type: densenet or lite')
    
    # Processing
    parser.add_argument('--resize', type=str, default='320x240',
                        help='Resize frames to WxH (should match training)')
    parser.add_argument('--close_threshold', type=float, default=0.5,
                        help='Threshold for predicting close (default: 0.5). '
                             'Lower value = more likely to predict close.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for inference (default: 16)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for files (default: same as video)')
    parser.add_argument('--skip_video', action='store_true',
                        help='Skip generating annotated video (only generate CSV)')
    
    # GPU
    parser.add_argument('--useGPU', type=str, default='True',
                        help='Use GPU if available')
    
    args = parser.parse_args()
    return args


def preprocess_frame(frame, resize, clahe):
    """
    Preprocess a video frame for model input.
    
    Args:
        frame: BGR frame from video
        resize: Tuple of (width, height) for resizing
        clahe: CLAHE object for preprocessing
    
    Returns:
        Preprocessed tensor ready for model
    """
    # Convert to grayscale
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    # Resize
    if resize is not None:
        gray = cv2.resize(gray, resize)
    
    # Gamma correction (same as training)
    table = 255.0 * (np.linspace(0, 1, 256) ** 0.8)
    gray = cv2.LUT(gray, table)
    
    # CLAHE
    gray = clahe.apply(np.uint8(gray))
    
    # Convert to PIL Image
    img = Image.fromarray(gray)
    
    # Transform (same as training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    tensor = transform(img)
    return tensor.unsqueeze(0)  # Add batch dimension


def draw_label_on_frame(frame, state, confidence):
    """
    Draw eye state label on the top-left corner of the frame.
    
    Args:
        frame: BGR frame
        state: 'open' or 'close'
        confidence: prediction confidence (0-1)
    
    Returns:
        Frame with label drawn
    """
    # Make a copy to avoid modifying original
    annotated = frame.copy()
    
    # Label text
    label = f"Eye: {state}"
    conf_text = f"({confidence:.1%})"
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    
    # Colors (BGR)
    if state == 'open':
        color = (0, 255, 0)  # Green for open
    else:
        color = (0, 0, 255)  # Red for close
    
    bg_color = (0, 0, 0)  # Black background
    
    # Get text size
    (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    (conf_w, conf_h), _ = cv2.getTextSize(conf_text, font, font_scale * 0.7, thickness)
    
    # Position (top-left corner with padding)
    x, y = 20, 40
    padding = 10
    
    # Draw background rectangle
    total_h = label_h + conf_h + padding * 3
    total_w = max(label_w, conf_w) + padding * 2
    cv2.rectangle(annotated, (x - padding, y - label_h - padding), 
                  (x + total_w, y + conf_h + padding * 2), bg_color, -1)
    
    # Draw main label
    cv2.putText(annotated, label, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    
    # Draw confidence
    cv2.putText(annotated, conf_text, (x, y + conf_h + padding), font, 
                font_scale * 0.7, (255, 255, 255), thickness - 1, cv2.LINE_AA)
    
    return annotated


def process_video(video_path, model, device, resize, close_threshold=0.5,
                  batch_size=16, skip_video=False, output_dir=None):
    """
    Process a video, predict eye states, and generate annotated video and CSV.
    
    Args:
        video_path: Path to input video file
        model: Trained model
        device: torch device
        resize: Tuple of (width, height) for model input
        close_threshold: Threshold for predicting close
        batch_size: Number of frames to process at once
        skip_video: If True, skip generating annotated video (only CSV)
        output_dir: Output directory (default: same as video)
    
    Returns:
        DataFrame with predictions
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video: {os.path.basename(video_path)}")
    print(f"  Resolution: {width}x{height}, FPS: {fps:.2f}")
    print(f"  Duration: {duration:.2f}s ({duration/60:.2f} min), Frames: {total_frames}")
    print(f"  Batch size: {batch_size}")
    
    # Determine output paths
    video_dir = os.path.dirname(os.path.abspath(video_path))
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    if output_dir is None:
        output_dir = video_dir
    os.makedirs(output_dir, exist_ok=True)
    
    output_video_path = os.path.join(output_dir, f"{video_name}_annotated.mp4")
    csv_path = os.path.join(output_dir, f"{video_name}_annotations.csv")
    
    # Setup video writer
    out = None
    if not skip_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        if not out.isOpened():
            raise ValueError(f"Cannot create output video: {output_video_path}")
        print(f"  Output video: {output_video_path}")
    else:
        print(f"  Video: SKIP (CSV only)")
    
    # CLAHE for preprocessing
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    
    # Process frames in batches
    results = []
    current_frame = 0
    
    model.eval()
    pbar = tqdm(total=total_frames, desc="Processing")
    
    with torch.no_grad():
        while current_frame < total_frames:
            # Read a batch of frames
            batch_frames = []
            batch_tensors = []
            batch_frame_nums = []
            
            for _ in range(batch_size):
                if current_frame >= total_frames:
                    break
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                batch_frames.append(frame)
                batch_frame_nums.append(current_frame)
                
                # Preprocess frame for model
                tensor = preprocess_frame(frame, resize, clahe)
                batch_tensors.append(tensor)
                
                current_frame += 1
            
            if not batch_tensors:
                break
            
            # Stack tensors and process as batch
            batch_tensor = torch.cat(batch_tensors, dim=0).to(device)
            outputs = model(batch_tensor)
            probs = torch.softmax(outputs, dim=1)
            
            # Process each result in the batch
            for i, (frame, frame_num) in enumerate(zip(batch_frames, batch_frame_nums)):
                time_sec = frame_num / fps
                prob_close = probs[i, 1].item()
                prob_open = probs[i, 0].item()
                
                # Decision based on threshold
                if prob_close > close_threshold:
                    state = 'close'
                    confidence = prob_close
                else:
                    state = 'open'
                    confidence = prob_open
                
                # Write annotated frame
                if out is not None:
                    annotated_frame = draw_label_on_frame(frame, state, confidence)
                    out.write(annotated_frame)
                
                results.append({
                    'frame': frame_num,
                    'time_sec': round(time_sec, 3),
                    'state': state,
                    'confidence': round(confidence, 4)
                })
            
            pbar.update(len(batch_frames))
    
    pbar.close()
    cap.release()
    if out is not None:
        out.release()
    
    # Create DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # Print statistics
    open_count = (df['state'] == 'open').sum()
    close_count = (df['state'] == 'close').sum()
    print(f"  Results: open={open_count}, close={close_count}")
    print(f"  Saved CSV: {csv_path}")
    if not skip_video:
        print(f"  Saved Video: {output_video_path}")
    
    return df


def main():
    args = parse_args()
    
    # Parse resize
    w, h = map(int, args.resize.split('x'))
    resize = (w, h)
    
    # Device setup
    use_gpu = args.useGPU.lower() == 'true' and torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {args.model}")
    if args.model_type == 'densenet':
        model = EyeStateClassifier(in_channels=1, num_classes=2)
    else:
        model = EyeStateClassifierLite(in_channels=1, num_classes=2)
    
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")
    
    # Check video exists
    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video file not found: {args.video}")
    
    # Process video
    print(f"\n{'='*60}")
    process_video(
        args.video,
        model,
        device,
        resize,
        close_threshold=args.close_threshold,
        batch_size=args.batch_size,
        skip_video=args.skip_video,
        output_dir=args.output_dir
    )
    
    print(f"\n{'='*60}")
    print("Done!")


if __name__ == '__main__':
    main()
