from pathlib import Path

def main():
    path = Path('src')
    for file in path.glob('**/*.py'):
        if file.name.startswith('__init__'):
            continue
        context = file.read_text()
        if 'from __future__ import annotations' not in context:
            print(file)
    

if __name__ == '__main__':
    main()
