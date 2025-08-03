#!/usr/bin/env python3
"""
Process episode scripts to extract lemma occurrences.

- Uses NLTK for better tokenization when needed
- Supports both raw and pre-normalized text
- Handles multi-word expressions
- Maintains accurate line numbers and positions
- Includes robust error handling
"""
import os
import re
import json
import argparse
from pathlib import Path
from collections import defaultdict

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Ensure NLTK is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt')

NLTK_AVAILABLE = True

class EpisodeProcessor:
    def __init__(self, use_nltk=True):
        """Initialize the processor.
        
        Args:
            use_nltk: Whether to use NLTK for tokenization (recommended)
        """
        self.use_nltk = use_nltk
        if not NLTK_AVAILABLE:
            raise RuntimeError("NLTK is required but not available. Please install it with: pip install nltk")
        if self.use_nltk:
            self._ensure_nltk_data()
    
    def _ensure_nltk_data(self):
        """Ensure required NLTK data is downloaded."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading required NLTK data...")
            nltk.download('punkt')
    
    def load_targets(self, targets_file='sb_targets.txt'):
        with open(targets_file, 'r') as f:
            return [line.strip().lower() for line in f if line.strip()]
    
    def extract_season_episode(self, filename):
        """Extract season and episode number from filename.
        
        Supports multiple filename formats:
        - S01E01
        - s01e01
        - Season 1/Episode 1
        - 1x01
        """
        filename = str(filename).lower()
        
        # Try different patterns
        patterns = [
            r's(\d+)e(\d+)',  # S01E01
            r'season[\s_]*(\d+).*?episode[\s_]*(\d+)',  # Season 1 Episode 1
            r'(\d+)x(\d+)',  # 1x01
            r'[^a-z](\d)(\d{2})[^a-z]'  # 101 (S1E01)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return int(match.group(1)), int(match.group(2))
        
        return None, None
    
    def process_text(self, text, lemmas):
        """Process text and find lemma occurrences.
        
        Args:
            text: Input text to process
            lemmas: List of lemmas to find
            
        Returns:
            Dictionary mapping lemmas to their occurrences
        """
        occurrences = {lemma: [] for lemma in lemmas}
        
        if self.use_nltk:
            # Use NLTK for better tokenization
            sentences = sent_tokenize(text)
            for sent in sentences:
                words = word_tokenize(sent.lower())
                for lemma in lemmas:
                    # Handle multi-word lemmas
                    if ' ' in lemma:
                        if lemma in ' '.join(words):
                            # Find all occurrences of multi-word lemma
                            for i in range(len(words) - len(lemma.split()) + 1):
                                if ' '.join(words[i:i+len(lemma.split())]) == lemma:
                                    occurrences[lemma].append({
                                        'token': lemma,
                                        'start': i,
                                        'end': i + len(lemma.split()),
                                        'sent': sent.strip()
                                    })
                    else:
                        # Single word lemma
                        for i, word in enumerate(words):
                            if lemma == word.lower():
                                occurrences[lemma].append({
                                    'token': word,
                                    'start': i,
                                    'end': i + 1,
                                    'sent': sent.strip()
                                })
        else:
            # Fallback to simple regex-based approach
            for line_num, line in enumerate(text.split('\n'), 1):
                line_lower = line.lower()
                for lemma in lemmas:
                    for match in re.finditer(re.escape(lemma), line_lower):
                        start, end = match.span()
                        token = line[start:end]
                        occurrences[lemma].append({
                            'token': token,
                            'start': start,
                            'end': end,
                            'line': line_num,
                            'sent': line.strip()
                        })
        
        return occurrences
    
    def process_episode(self, filepath, lemmas):
        """Process a single episode file.
        
        Args:
            filepath: Path to the episode file
            lemmas: List of lemmas to find
            
        Returns:
            Tuple of (season, episode, occurrences)
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            filename = os.path.basename(filepath)
            season, episode = self.extract_season_episode(filename)
            
            if season is None or episode is None:
                print(f"Warning: Could not parse season/episode from {filename}")
                return None, None, {}
            
            occurrences = self.process_text(content, lemmas)
            return season, episode, occurrences
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return None, None, {}
    
    def process_directory(self, input_dir, lemmas, output_dir='data'):
        """Process all episodes in a directory.
        
        Args:
            input_dir: Directory containing episode files
            lemmas: List of lemmas to find
            output_dir: Directory to save output files
            
        Returns:
            Dictionary of results by season and lemma
        """
        results = defaultdict(lambda: defaultdict(list))
        
        # Find all text files in the directory
        text_files = []
        for ext in ('*.txt', '*.srt'):
            text_files.extend(Path(input_dir).rglob(ext))
        
        # Process each file
        for filepath in text_files:
            season, episode, occurrences = self.process_episode(filepath, lemmas)
            if season is not None and episode is not None:
                for lemma, lemma_occurrences in occurrences.items():
                    for occ in lemma_occurrences:
                        occ.update({
                            'episode': f"S{season:02d}E{episode:02d}",
                            'file': str(filepath.relative_to(input_dir))
                        })
                        results[season][lemma].append(occ)
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        for season, lemmas_data in results.items():
            season_dir = os.path.join(output_dir, f'season{season}')
            os.makedirs(season_dir, exist_ok=True)
            
            for lemma, occurrences in lemmas_data.items():
                output_file = os.path.join(season_dir, f"{lemma.replace(' ', '_')}.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(occurrences, f, indent=2, ensure_ascii=False)
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Process episode scripts to extract lemma occurrences.')
    parser.add_argument('input_dir', help='Directory containing episode files')
    parser.add_argument('--output-dir', default='data', help='Output directory for results')
    parser.add_argument('--targets-file', default='sb_targets.txt', help='File containing target lemmas')
    parser.add_argument('--no-nltk', action='store_true', help='Disable NLTK processing')
    
    args = parser.parse_args()
    
    processor = EpisodeProcessor(use_nltk=not args.no_nltk)
    lemmas = processor.load_targets(args.targets_file)
    
    print(f"Processing episodes in: {args.input_dir}")
    print(f"Looking for {len(lemmas)} lemmas")
    print(f"Using NLTK: {processor.use_nltk}")
    
    results = processor.process_directory(args.input_dir, lemmas, args.output_dir)
    
    # Print summary
    total_episodes = sum(len(lemmas) for lemmas in results.values())
    print(f"\nProcessed {sum(len(episodes) for episodes in results.values())} episodes in {len(results)} seasons")
    
    for season, lemmas_data in sorted(results.items()):
        print(f"\nSeason {season}:")
        for lemma, occurrences in sorted(lemmas_data.items()):
            print(f"  {lemma}: {len(occurrences)} occurrences")

if __name__ == '__main__':
    main()
