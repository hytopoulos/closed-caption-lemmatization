#!/usr/bin/env python3
"""
Plot the occurrences of lemmas across seasons.

This script reads the lemma data from the data directory and creates
visualizations showing how often each lemma appears in each season, with
options for raw counts or term frequency normalization.
"""
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from pathlib import Path

def count_tokens_in_season(season_path):
    """Count total tokens in a season for normalization."""
    token_counter = Counter()
    
    # Process each lemma file in the season
    for lemma_file in os.listdir(season_path):
        if not lemma_file.endswith('.json'):
            continue
            
        file_path = os.path.join(season_path, lemma_file)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                occurrences = json.load(f)
                # Count all tokens in this lemma's occurrences
                for occ in occurrences:
                    if 'sent' in occ:
                        # Simple word count for normalization
                        words = occ['sent'].split()
                        token_counter.update(words)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error counting tokens in {file_path}: {e}")
    
    return sum(token_counter.values())

def load_lemma_data(data_dir='data', normalize=False):
    """Load lemma data from JSON files in the specified directory.
    
    Args:
        data_dir: Directory containing the lemma data
        normalize: If True, normalize counts by total tokens in season
        
    Returns:
        Tuple of (data, season_totals) where data is a nested dict of lemma counts
        and season_totals is a dict of total tokens per season
    """
    data = defaultdict(lambda: defaultdict(float))
    season_totals = {}
    
    # Get all season directories
    season_dirs = sorted(
        [d for d in os.listdir(data_dir) if d.startswith('season') and os.path.isdir(os.path.join(data_dir, d))],
        key=lambda x: int(x[6:])  # Sort by season number
    )
    
    # First pass: count total tokens in each season if normalizing
    if normalize:
        for season_dir in season_dirs:
            season_num = int(season_dir[6:])
            season_path = os.path.join(data_dir, season_dir)
            season_totals[season_num] = count_tokens_in_season(season_path)
    
    # Second pass: load and count lemma occurrences
    for season_dir in season_dirs:
        season_num = int(season_dir[6:])
        season_path = os.path.join(data_dir, season_dir)
        
        # Process each lemma file in the season
        for lemma_file in os.listdir(season_path):
            if not lemma_file.endswith('.json'):
                continue
                
            lemma = os.path.splitext(lemma_file)[0].replace('_', ' ')
            file_path = os.path.join(season_path, lemma_file)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    occurrences = json.load(f)
                    count = len(occurrences)
                    
                    # Normalize by total tokens in season if requested
                    if normalize and season_num in season_totals and season_totals[season_num] > 0:
                        data[lemma][season_num] = (count / season_totals[season_num]) * 1000  # Per 1000 tokens
                    else:
                        data[lemma][season_num] = count
                        
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error loading {file_path}: {e}")
    
    return data, season_totals

def plot_lemma_occurrences(data, output_file='lemma_occurrences.png', normalize=False):
    """Plot the occurrences of each lemma across seasons.
    
    Args:
        data: Dictionary mapping lemmas to season counts
        output_file: Path to save the output plot
        normalize: Whether the data is normalized (affects y-axis label)
    """
    if not data:
        print("No data to plot!")
        return
    
    # Prepare data for plotting
    seasons = sorted({s for lemma_data in data.values() for s in lemma_data})
    
    # Sort lemmas by total frequency (descending)
    lemma_totals = {lemma: sum(season_data.values()) for lemma, season_data in data.items()}
    lemmas = sorted(data.keys(), key=lambda x: lemma_totals[x], reverse=True)
    
    # Create a 2D array of counts
    counts = np.zeros((len(lemmas), len(seasons)))
    
    for i, lemma in enumerate(lemmas):
        for j, season in enumerate(seasons):
            counts[i, j] = data[lemma].get(season, 0)
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Create a colormap for better distinction between lemmas
    # Use a combination of colormaps for maximum contrast
    cmap1 = plt.get_cmap('tab20c', 20)
    cmap2 = plt.get_cmap('Set3', 12)
    
    # Plot each lemma as a line
    for i, lemma in enumerate(lemmas):
        # Alternate between colormaps for better contrast
        if i < 20:
            color = cmap1(i)
        elif i < 32:
            color = cmap2(i - 20)
        else:
            # For more than 32 lemmas, cycle through both colormaps with offset
            if i % 2 == 0:
                color = cmap1((i // 2) % 20)
            else:
                color = cmap2(((i - 1) // 2) % 12)
        
        plt.plot(
            seasons, 
            counts[i], 
            marker='o', 
            label=lemma.capitalize(),
            color=color,
            linewidth=2.5
        )
    
    # Customize the plot
    ylabel = 'Occurrences per 1000 tokens' if normalize else 'Number of Occurrences'
    title = 'Normalized Lemma Occurrences Across Seasons' if normalize else 'Lemma Occurrences Across Seasons'
    
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('Season', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(seasons)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend outside the plot
    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.,
        title='Lemmas',
        title_fontsize='large',
        ncol=2 if len(lemmas) > 10 else 1  # Two columns for many lemmas
    )
    
    # Adjust layout to prevent cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()  # Close the figure to free memory

def plot_heatmap(data, output_file='lemma_heatmap.png', normalize=False):
    """Create a heatmap of lemma occurrences by season.
    
    Args:
        data: Dictionary mapping lemmas to season counts
        output_file: Path to save the output plot
        normalize: Whether the data is normalized (affects colorbar label)
    """
    if not data:
        print("No data to plot!")
        return
    
    # Prepare data for heatmap
    lemmas = sorted(data.keys())
    seasons = sorted({s for lemma_data in data.values() for s in lemma_data})
    
    # Create a 2D array of counts
    counts = np.zeros((len(lemmas), len(seasons)))
    
    for i, lemma in enumerate(lemmas):
        for j, season in enumerate(seasons):
            counts[i, j] = data[lemma].get(season, 0)
    
    # Create the heatmap
    plt.figure(figsize=(14, 18))
    
    # Create heatmap with colorbar
    im = plt.imshow(counts, cmap='YlOrRd', aspect='auto')
    
    # Add colorbar with appropriate label
    cbar_label = 'Occurrences per 1000 tokens' if normalize else 'Number of Occurrences'
    cbar = plt.colorbar(im, pad=0.02)
    cbar.set_label(cbar_label, rotation=270, labelpad=15)
    
    # Customize the plot
    title = 'Normalized Lemma Occurrences by Season' if normalize else 'Lemma Occurrences by Season'
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('Season', fontsize=12)
    plt.ylabel('Lemma', fontsize=12)
    
    # Set tick labels
    plt.xticks(np.arange(len(seasons)), [f'S{s}' for s in seasons])
    plt.yticks(np.arange(len(lemmas)), [l.capitalize() for l in lemmas])
    
    # Add text annotations (only for non-normalized or when values are integers)
    if not normalize or all(x == int(x) for row in counts for x in row):
        for i in range(len(lemmas)):
            for j in range(len(seasons)):
                val = counts[i, j]
                if val > 0:  # Only show non-zero values
                    # Format based on whether it's normalized
                    if normalize:
                        text = f"{val:.1f}" if val < 10 else f"{int(val)}"
                    else:
                        text = str(int(val))
                    
                    # Choose text color based on background brightness
                    text_color = 'black' if val < np.max(counts)/2 else 'white'
                    plt.text(j, i, text,
                            ha="center", va="center",
                            color=text_color, fontsize=8)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {output_file}")
    plt.close()  # Close the figure to free memory

def main():
    parser = argparse.ArgumentParser(description='Plot lemma occurrences across seasons.')
    parser.add_argument('--data-dir', default='data', help='Directory containing lemma data')
    parser.add_argument('--output-dir', default='plots', help='Directory to save output plots')
    parser.add_argument('--normalize', action='store_true', 
                       help='Normalize by term frequency (occurrences per 1000 tokens)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the data
    print(f"Loading lemma data from {args.data_dir}...")
    data, season_totals = load_lemma_data(args.data_dir, normalize=args.normalize)
    
    if not data:
        print("No data found! Make sure to run process_episodes.py first.")
        return
    
    # Print season totals for reference
    if args.normalize:
        print("\nTotal tokens per season:")
        for season, total in sorted(season_totals.items()):
            print(f"  Season {season}: {total:,} tokens")
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Determine output filenames based on normalization
    if args.normalize:
        line_plot = os.path.join(args.output_dir, 'normalized_lemma_occurrences.png')
        heatmap = os.path.join(args.output_dir, 'normalized_lemma_heatmap.png')
    else:
        line_plot = os.path.join(args.output_dir, 'lemma_occurrences.png')
        heatmap = os.path.join(args.output_dir, 'lemma_heatmap.png')
    
    # Generate plots
    plot_lemma_occurrences(
        data, 
        output_file=line_plot,
        normalize=args.normalize
    )
    
    plot_heatmap(
        data,
        output_file=heatmap,
        normalize=args.normalize
    )
    
    print("\nDone!")
    print(f"Plots saved to: {os.path.abspath(args.output_dir)}")

if __name__ == '__main__':
    main()
