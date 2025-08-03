import os
import re
from pathlib import Path

def parse_srt_file(file_path):
    """Parse an SRT file and return the dialogue text."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Remove timestamps and SRT formatting
        # This regex removes the numeric counter and timestamps (e.g., "1\n00:00:00,000 --> 00:00:02,500\n")
        content = re.sub(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n', '', content)
        
        # Remove any remaining HTML tags
        content = re.sub(r'<[^>]+>', '', content)
        
        # Remove any remaining SRT formatting
        content = re.sub(r'\{.*?\}', '', content)  # Remove text within {}
        content = re.sub(r'\[.*?\]', '', content)  # Remove text within []
        
        # Remove multiple newlines and leading/trailing whitespace
        content = ' '.join(content.split())
        
        return content
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return ""

def split_into_sentences(text):
    """Split text into sentences using a simple approach."""
    if not text:
        return []
        
    # First, split on sentence end markers
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Further split sentences that might have been missed
    result = []
    for sentence in sentences:
        # Split on common dialogue patterns
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\(\[])', sentence)
        result.extend(parts)
    
    # Clean up and filter empty sentences
    return [s.strip() for s in result if s.strip()]

def normalize_file(input_file, output_file):
    """Normalize a single caption file to have one sentence per line"""
    try:
        # Parse the SRT file to extract dialogue
        content = parse_srt_file(input_file)
        if not content:
            print(f"Skipping empty or invalid file: {input_file}")
            return False
            
        # Split into sentences
        sentences = split_into_sentences(content)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Write each sentence on a new line
        with open(output_file, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                # Remove any remaining extra whitespace and write the sentence
                cleaned = sentence.strip()
                if cleaned:  # Only write non-empty lines
                    f.write(cleaned + '\n')
        return True
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return False

def process_directory(input_dir, output_dir):
    """Process all script files in the input directory and its subdirectories"""
    # Find all .srt files in the input directory and its subdirectories
    input_files = []
    for ext in ('*.srt', '*.SRT'):
        input_files.extend(Path(input_dir).rglob(ext))
    
    if not input_files:
        print(f"No .txt files found in {input_dir}")
        return
    
    print(f"Found {len(input_files)} files to process...")
    
    success_count = 0
    
    for input_file in input_files:
        # Convert to string if it's a Path object
        input_file = str(input_file)
        
        # Get the relative path to maintain directory structure
        rel_path = os.path.relpath(input_file, input_dir)
        
        # Change the extension to .txt for the output file
        output_file = os.path.splitext(os.path.join(output_dir, rel_path))[0] + '.txt'
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        print(f"Processing: {rel_path}")
        
        if normalize_file(input_file, output_file):
            success_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {success_count}/{len(input_files)} files")

def main():
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python normalize_captions.py <input_dir> <output_dir>")
        print("Example: python normalize_captions.py scripts normalized_scripts")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    print(f"Normalizing scripts from: {input_dir}")
    print(f"Saving normalized scripts to: {output_dir}")
    print("-" * 50)
    
    process_directory(input_dir, output_dir)
    
    print("\nAll done! Check the output directory for the normalized scripts.")
    print("Normalization complete!")

if __name__ == "__main__":
    main()
