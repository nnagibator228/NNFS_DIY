import os

def delete_comments(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File '{file_path}' does not exist.")
    with open(file_path, 'r') as file:
        lines = file.readlines()
        deleted_comments = []
        for line in lines:
            if line.strip().startswith('#'):
                pass
            else:
                deleted_comments.append(line)

    with open(file_path, 'w') as file:
        file.writelines(deleted_comments)

    print(f"Comments in '{file_path}' have been deleted.")


# Usage example
file_path = 'main.py'
delete_comments(file_path)