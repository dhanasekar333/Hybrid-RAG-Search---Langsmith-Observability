import os
from rag import FILES_FOLDER, SUPPORTED, get_hash, save_hash   # replace "rag" with your script's file name (without .py)

for name in os.listdir(FILES_FOLDER):
    if name.lower().endswith(SUPPORTED):
        save_hash(name, get_hash(os.path.join(FILES_FOLDER, name)))
        print("Saved hash for", name)