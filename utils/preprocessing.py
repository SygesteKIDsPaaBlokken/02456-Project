from tqdm import tqdm

def remove_stop_words(passages, stop_words: list[str] = None):
    if stop_words is None:
        import nltk
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
    
    passages_clean = []
    for passage in tqdm(passages, desc='Removing stop words'):  # ~20
        temp = ''
        for word in passage.split(' '):
            if word not in stop_words:
                temp += word + ' '
        passages_clean.append(temp[:-1])

    return passages_clean