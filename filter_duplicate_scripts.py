#!/usr/bin/env python3
"""
Filter out duplicate caption files using semantic similarity.

This script processes .srt files from the scripts directory,
uses sentence transformers to generate embeddings, and identifies
near-duplicate files based on cosine similarity.
"""
import os
import re
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Configuration
SIMILARITY_THRESHOLD = 0.95  # Files with similarity above this will be considered duplicates
BATCH_SIZE = 32  # Number of files to process in each batch
MODEL_NAME = 'all-MiniLM-L6-v2'  # Lightweight but effective model for this task

def load_srt_file(file_path: Path) -> str:
    """Load and clean the content of an SRT file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Remove timestamps and SRT formatting
        content = re.sub(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n', '', content)
        content = re.sub(r'<[^>]+>', '', content)  # Remove HTML tags
        content = re.sub(r'\{.*?\}', '', content)  # Remove text within {}
        content = re.sub(r'\[.*?\]', '', content)  # Remove text within []
        
        # Remove multiple spaces and newlines
        content = ' '.join(content.split())
        return content
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def get_episode_info(file_path: Path) -> Tuple[str, str, str]:
    """Extract season and episode information from the file path."""
    # Try to extract season and episode numbers from the path
    path_str = str(file_path).lower()
    season_match = re.search(r'season[\s_](\d+)', path_str)
    episode_match = re.search(r'episode[\s_](\d+)', path_str)
    
    # Fallback to filename patterns if not found in path
    if not season_match or not episode_match:
        file_name = file_path.name.lower()
        season_match = season_match or re.search(r's(\d+)e\d+', file_name) or re.search(r's(\d+)\.', file_name)
        episode_match = episode_match or re.search(r's\d+e(\d+)', file_name) or re.search(r'e(\d+)\.', file_name)
    
    season = season_match.group(1).zfill(2) if season_match else '00'
    episode = episode_match.group(1).zfill(2) if episode_match else '00'
    
    return season, episode, file_path.stem

def find_duplicates(files: List[Path], model) -> Dict[str, List[Path]]:
    """Find duplicate files based on semantic similarity of their content."""
    print(f"Processing {len(files)} files to find duplicates...")
    
    # Group files by season and episode to reduce comparison space
    file_groups = defaultdict(list)
    for file_path in files:
        season, episode, _ = get_episode_info(file_path)
        file_groups[(season, episode)].append(file_path)
    
    duplicates = defaultdict(list)
    
    # Process each group separately
    for (season, episode), group_files in tqdm(file_groups.items(), desc="Processing groups"):
        if len(group_files) < 2:
            continue  # No potential duplicates in this group
            
        # Load and clean file contents
        contents = [load_srt_file(f) for f in group_files]
        
        # Skip empty files
        valid_indices = [i for i, c in enumerate(contents) if c.strip()]
        valid_files = [group_files[i] for i in valid_indices]
        valid_contents = [contents[i] for i in valid_indices]
        
        if len(valid_contents) < 2:
            continue
        
        # Generate embeddings
        embeddings = model.encode(valid_contents, convert_to_tensor=True, show_progress_bar=False)
        
        # Calculate cosine similarity between all pairs
        cos_scores = util.cos_sim(embeddings, embeddings).cpu().numpy()
        
        # Find duplicates
        processed = set()
        for i in range(len(valid_contents)):
            if i in processed:
                continue
                
            # Find all files similar to this one
            similar_indices = np.where(cos_scores[i] > SIMILARITY_THRESHOLD)[0]
            similar_indices = [j for j in similar_indices if j != i]
            
            if similar_indices:
                original_file = valid_files[i]
                duplicate_files = [valid_files[j] for j in similar_indices]
                
                # Keep the shortest filename as the original (usually the cleanest version)
                all_files = [original_file] + duplicate_files
                all_files.sort(key=lambda x: len(x.name))
                
                duplicates[str(all_files[0])] = [str(f) for f in all_files[1:]]
                processed.update([i] + similar_indices)
    
    return duplicates

def filter_duplicates():
    """Main function to filter duplicate caption files."""
    # Initialize the model
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    
    # Find all SRT files
    input_dir = Path("scripts")
    output_dir = Path("filtered_scripts")
    
    srt_files = list(input_dir.rglob("*.srt")) + list(input_dir.rglob("*.SRT"))
    print(f"Found {len(srt_files)} SRT files in {input_dir}")
    
    if not srt_files:
        print("No SRT files found. Exiting.")
        return
    
    # Find duplicates
    duplicates = find_duplicates(srt_files, model)
    
    # Create output directory structure and copy non-duplicate files
    files_to_keep = set()
    files_to_remove = set()
    
    # First, identify all files to keep and remove
    for original, dup_list in duplicates.items():
        files_to_keep.add(original)
        files_to_remove.update(dup_list)
    
    # Add all non-duplicate files to the keep list
    for file_path in srt_files:
        if str(file_path) not in files_to_remove and str(file_path) not in files_to_keep:
            files_to_keep.add(str(file_path))
    
    print(f"\nFound {len(duplicates)} sets of duplicate files.")
    print(f"Keeping {len(files_to_keep)} files, removing {len(files_to_remove)} duplicates.")
    
    # Create output directory and copy files
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy all files to keep to the output directory
    for src_path in tqdm(files_to_keep, desc="Copying files"):
        src_path = Path(src_path)
        rel_path = src_path.relative_to(input_dir)
        dest_path = output_dir / rel_path
        
        # Create parent directories if they don't exist
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy the file
        shutil.copy2(src_path, dest_path)
    
    # Save the duplicate information to a JSON file
    duplicates_file = output_dir / "duplicates_info.json"
    with open(duplicates_file, 'w', encoding='utf-8') as f:
        json.dump({
            'duplicates': {k: v for k, v in duplicates.items()},
            'total_files_processed': len(srt_files),
            'files_kept': len(files_to_keep),
            'duplicates_removed': len(files_to_remove),
            'similarity_threshold': SIMILARITY_THRESHOLD
        }, f, indent=2)
    
    print(f"\nFiltering complete!")
    print(f"Kept {len(files_to_keep)} files in {output_dir}")
    print(f"Removed {len(files_to_remove)} duplicate files")
    print(f"Duplicate information saved to: {duplicates_file}")

if __name__ == "__main__":
    filter_duplicates()
