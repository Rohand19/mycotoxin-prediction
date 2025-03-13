#!/usr/bin/env python3
"""Script to fix common issues in the codebase."""

import os
import re
import glob

def fix_trailing_whitespace(file_path):
    """Fix trailing whitespace in a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix trailing whitespace
    fixed_content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)
    
    # Ensure file ends with a newline
    if not fixed_content.endswith('\n'):
        fixed_content += '\n'
    
    if content != fixed_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        return True
    return False

def fix_blank_lines(file_path):
    """Fix blank lines with whitespace in a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix blank lines with whitespace
    fixed_content = re.sub(r'^[ \t]+$', '', content, flags=re.MULTILINE)
    
    if content != fixed_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        return True
    return False

def fix_imports(file_path):
    """Fix unused imports in a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add numpy import if np is used but not imported
    if 'np.' in content and 'import numpy as np' not in content:
        fixed_content = re.sub(
            r'(import .*?\n\n)',
            r'\1import numpy as np\n\n',
            content,
            count=1
        )
        
        if content != fixed_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            return True
    return False

def fix_all_files():
    """Fix common issues in all Python files."""
    python_files = glob.glob('src/**/*.py', recursive=True)
    
    fixed_files = 0
    for file_path in python_files:
        fixed_ws = fix_trailing_whitespace(file_path)
        fixed_bl = fix_blank_lines(file_path)
        fixed_im = fix_imports(file_path)
        
        if fixed_ws or fixed_bl or fixed_im:
            fixed_files += 1
            print(f"Fixed issues in {file_path}")
    
    print(f"\nFixed issues in {fixed_files} files")

if __name__ == "__main__":
    fix_all_files() 