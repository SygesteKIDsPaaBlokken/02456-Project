import numpy as np
import pandas as pd
import fasttext.util


if __name__ == '__main__':
    fasttext.util.download_model('en', if_exists='ignore')  # English
    ft = fasttext.load_model('cc.en.300.bin')
    
    pass