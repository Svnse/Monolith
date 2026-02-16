import os

def combine_files_to_txt(output_filename="monolith_source.txt"):
    # Folders to ignore to keep the txt clean
    ignore_folders = {'.git', '__pycache__', '.venv', 'node_modules', '.idea'}
    # Files to ignore (including the output file itself)
    ignore_files = {output_filename, os.path.basename(__file__)}

    with open(output_filename, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk('.'):
            # Prune ignored directories
            dirs[:] = [d for d in dirs if d not in ignore_folders]
            
            for filename in files:
                if filename in ignore_files:
                    continue
                
                file_path = os.path.join(root, filename)
                
                # Write a clear header for each file
                outfile.write(f"\n{'='*80}\n")
                outfile.write(f"FILE: {file_path}\n")
                outfile.write(f"{'='*80}\n\n")
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                        outfile.write(infile.read())
                except Exception as e:
                    outfile.write(f"--- ERROR READING FILE: {e} ---\n")
                
                outfile.write("\n\n")

    print(f"Successfully combined files into {output_filename}")

if __name__ == "__main__":
    combine_files_to_txt()