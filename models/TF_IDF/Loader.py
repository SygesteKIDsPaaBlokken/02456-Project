###################################################
# From https://github.com/alchemistwu/machine_learning_notebook/tree/main/examples/information_retrieval
###################################################

import re
import string

def clean_text(text: str) -> str:
    """
    Toy method for cleaning text
    Args:
        text (str): dirty text

    Returns:
        str: cleaned text
    """
    # Remove Unicode
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Remove Mentions
    text = re.sub(r'@\w+', '', text)
    # Lowercase the document
    text = text.lower()
    # Remove punctuations
    text = re.sub(r'[%s]' % re.escape(string.punctuation),
                  ' ',
                  text)
    # Remove numbers
    text = re.sub(r'[0-9]', '', text)
    # Remove the doubled space
    text = re.sub(r'\s{2,}', ' ', text)
    return text