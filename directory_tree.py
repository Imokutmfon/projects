import os

def print_directory_tree(startpath):
    """
    Prints the directory structure of the given path in a tree-like format.
    Limits the number of files displayed per directory to 3 to avoid clutter.
    """
    print(f"Directory structure for: {startpath}")
    
    for root, dirs, files in os.walk(startpath):
        # Calculate depth to handle indentation
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        
        # Print the current directory name
        print(f'{indent}{os.path.basename(root)}/')
        
        # Prepare the indentation for files inside this directory
        subindent = ' ' * 4 * (level + 1)
        
        # specific logic to limit file output
        file_limit = 3
        
        for f in files[:file_limit]:
            print(f'{subindent}{f}')
            
        # If there are more files than the limit, show a summary count
        if len(files) > file_limit:
            print(f'{subindent}... ({len(files) - file_limit} more files)')

# Kaggle specific path
# You can change this to '.' to list current directory files
input_path = '/kaggle/input'

# Check if path exists (useful if testing locally vs on Kaggle)
if os.path.exists(input_path):
    print_directory_tree(input_path)
else:
    print(f"Path '{input_path}' not found. Printing current directory structure instead:\n")
    print_directory_tree('.')
