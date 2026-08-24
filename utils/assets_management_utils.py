import os
import json
import hashlib
import random
import torch

from pathlib import Path

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUTS_PATH = f"{BASE_PATH}/outputs"




def generate_8char_tag(dictionary_object):

    """
    Generate a unique 8-character alphanumeric ID for a given dictionary object.

    Inputs:
        dictionary_object (dict): The dictionary for which to generate the ID.
    Outputs:
        tag (str): An 8-character alphanumeric ID.
    """

    chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    tag = ""

    json_str = json.dumps(dictionary_object, sort_keys=True, separators=(',', ':'))
    sha256_hash = hashlib.sha256(json_str.encode('utf-8')).hexdigest()
    
    #Take the first 12 hex characters 
    partial_hex = sha256_hash[:12]
    hash_int = int(partial_hex, 16)
    
    #Loop exactly 8 times to build an 8-character ID
    for _ in range(8):
        hash_int, remainder = divmod(hash_int, 62)
        tag = chars[remainder] + tag
        
    return tag


def key_management(MANIFEST, dictionary_key, mode = "query", object = None):

    """ 
    function for managing a manifest file containing dictionary keys and their corresponding objects.

    Inputs:
        MANIFEST (str): Path to the manifest JSON file.
        dictionary_key (str): The key to query, load, or save in the manifest.
        mode (str): Operation mode. Can be "query", "load", or "save".
        object (any): The object to save in the manifest when mode is "save". Ignored for "query" and "load" modes.
    Outputs:
        For "query" mode: Returns True if the key exists, False otherwise.
        For "load" mode: Returns the object associated with the key if it exists, None otherwise.
        For "save" mode: Saves the object under the specified key in the manifest.
    """

    MANIFEST = Path(MANIFEST)

    if mode in ["query", "load"]:
        if MANIFEST.exists() and MANIFEST.stat().st_size > 0:
            with open(MANIFEST, "r") as file:
                manifest = json.load(file)
            if dictionary_key in manifest:
                if mode == "query":
                    return True
                elif mode == "load":
                    return manifest[dictionary_key]
        return False if mode == "query" else None
    
    elif mode == "save":
        if not MANIFEST.exists() or MANIFEST.stat().st_size == 0:
            os.makedirs(MANIFEST.parent, exist_ok=True)
            manifest = {}
        else:
            with open(MANIFEST, "r") as file:
                manifest = json.load(file)

        if object is None:
            raise ValueError("Object must be provided when saving. Signature is: key_management(MANIFEST, dictionary_key, object=object, mode='save')")
        manifest[dictionary_key] = object
        with open(MANIFEST, "w") as file:
            json.dump(manifest, file, separators=(',', ':'), indent=4, cls=CompactListEncoder)







class CompactListEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that outputs lists in a compact format (single line) while maintaining the default behavior for other data types.
    """
    def iterencode(self, o, _one_shot=False):
        # If it's a list and all elements are numbers (int/float), make it a single line
        if isinstance(o, list) and all(isinstance(x, (int, float)) for x in o):
            yield json.dumps(o)
        elif isinstance(o, dict):
            # For dicts, we want to maintain the indent behavior
            yield '{\n'
            indent_str = ' ' * self.indent if self.indent else ''
            
            items = []
            for k, v in o.items():
                key_str = json.dumps(k)
                # Recursively call iterencode for the values
                val_chunks = list(self.iterencode(v))
                # Join the chunks and indent them if they span multiple lines
                val_str = "".join(val_chunks)
                if '\n' in val_str:
                    val_str = val_str.replace('\n', '\n' + indent_str)
                items.append(f"{indent_str}{key_str}: {val_str}")
                
            yield ',\n'.join(items)
            yield '\n}'
        else:
            yield from super().iterencode(o, _one_shot)




def set_random_seeds(seed: int):
    """Sets the random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)