#!/usr/bin/env python3
"""
Perform TF-IDF analysis on normalized episode scripts.

This script reads all text files from the normalized_scripts directory,
performs TF-IDF analysis, and outputs the top 100 terms with their scores.
"""
import re
import nltk
import argparse
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer


def load_episode_scripts(scripts_dir='normalized_scripts'):
    """Load all episode scripts from the specified directory.
    
    Args:
        scripts_dir: Directory containing the normalized scripts
        
    Returns:
        List of tuples (episode_name, script_text)
    """
    episodes = []
    
    # Walk through all season directories
    for season_dir in Path(scripts_dir).iterdir():
        if not season_dir.is_dir():
            continue
            
        # Walk through all episode directories
        for episode_dir in season_dir.iterdir():
            if not episode_dir.is_dir():
                continue
                
            # Process all text files in the episode directory
            for script_file in episode_dir.iterdir():
                if script_file.suffix == '.txt':
                    try:
                        with open(script_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            episode_name = f"{season_dir.name}/{episode_dir.name}/{script_file.name}"
                            episodes.append((episode_name, content))
                    except Exception as e:
                        print(f"Error reading {script_file}: {e}")
    
    return episodes


def preprocess_text(text):
    """Preprocess text for TF-IDF analysis.
    
    Args:
        text: Raw text to preprocess
        
    Returns:
        Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove non-alphabetic characters and extra whitespace
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Tokenize and remove stopwords
    tokens = text.split()
    
    # Download stopwords if needed
    try:
        stop_words = set(nltk.corpus.stopwords.words('english'))
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords')
        stop_words = set(nltk.corpus.stopwords.words('english'))
    
    # Filter out stopwords and short words
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    return ' '.join(tokens)


def perform_tfidf_analysis(episodes, top_n=100):
    """Perform TF-IDF analysis on episode scripts.
    
    Args:
        episodes: List of tuples (episode_name, script_text)
        top_n: Number of top terms to return
        
    Returns:
        List of tuples (term, score) for top N terms
    """
    if not episodes:
        print("No episodes found!")
        return []
    
    print(f"Processing {len(episodes)} episodes...")
    
    # Preprocess all episode texts
    episode_names, episode_texts = zip(*episodes)
    processed_texts = [preprocess_text(text) for text in episode_texts]
    
    # Perform TF-IDF analysis
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    
    # Get feature names and their mean TF-IDF scores
    feature_names = vectorizer.get_feature_names_out()
    mean_scores = tfidf_matrix.mean(axis=0).A1
    
    # Create list of (term, score) tuples and sort by score
    term_scores = list(zip(feature_names, mean_scores))
    term_scores.sort(key=lambda x: x[1], reverse=True)
    
    return term_scores[:top_n]


def main():
    parser = argparse.ArgumentParser(description='Perform TF-IDF analysis on episode scripts.')
    parser.add_argument('--scripts-dir', default='normalized_scripts', help='Directory containing normalized scripts')
    parser.add_argument('--top-n', type=int, default=100, help='Number of top terms to output')
    parser.add_argument('--output-file', default='top100.txt', help='Output file for top terms')
    
    args = parser.parse_args()
    
    # Load episode scripts
    print(f"Loading episode scripts from {args.scripts_dir}...")
    episodes = load_episode_scripts(args.scripts_dir)
    
    if not episodes:
        print("No episodes found! Make sure the directory structure is correct.")
        return
    
    # Perform TF-IDF analysis
    print("Performing TF-IDF analysis...")
    top_terms = perform_tfidf_analysis(episodes, args.top_n)
    
    # Output results to console
    print(f"\nTop {len(top_terms)} terms by TF-IDF score:")
    print("-" * 40)
    for i, (term, score) in enumerate(top_terms, 1):
        print(f"{i:2d}. {term:<20} {score:.4f}")
    
    # Output terms to file (one per line)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for term, score in top_terms:
            f.write(f"{term}\n")
    
    print(f"\nTop terms also saved to {args.output_file}")


if __name__ == '__main__':
    main()
