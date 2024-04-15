import os
import ast

def extract_imports(file_path):
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read(), filename=file_path)

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
    return imports

def summarize_packages_in_directory(directory):
    all_packages = set()
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                imports = extract_imports(file_path)
                all_packages.update(imports)
    return sorted(all_packages)

if __name__ == "__main__":
    directory = os.getcwd()  # Use the current directory
    packages = summarize_packages_in_directory(directory)
    print("Required packages:")
    for package in packages:
        print(package)
