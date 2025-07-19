import random
import os
from typing import Optional


def fetch_positive_prompt() -> Optional[str]:
    """
    Reads all positive prompts from the file and returns a random one.
    
    Returns:
        str: A random positive prompt, or None if file not found or empty
    """
    try:
        # Get the directory where this script is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'random_prompts', 'positive_prompts.txt')
        
        # Read all prompts from the file
        with open(file_path, 'r', encoding='utf-8') as file:
            prompts = [line.strip() for line in file if line.strip()]
        
        # Return a random prompt if any exist
        if prompts:
            return random.choice(prompts)
        else:
            return None
            
    except FileNotFoundError:
        print(f"Error: positive_prompts.txt not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading positive prompts: {e}")
        return None


def fetch_negative_prompt() -> Optional[str]:
    """
    Reads all negative prompts from the file and returns a random one.
    
    Returns:
        str: A random negative prompt, or None if file not found or empty
    """
    try:
        # Get the directory where this script is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'random_prompts', 'negative_prompts.txt')
        
        # Read all prompts from the file
        with open(file_path, 'r', encoding='utf-8') as file:
            prompts = [line.strip() for line in file if line.strip()]
        
        # Return a random prompt if any exist
        if prompts:
            return random.choice(prompts)
        else:
            return None
            
    except FileNotFoundError:
        print(f"Error: negative_prompts.txt not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading negative prompts: {e}")
        return None
