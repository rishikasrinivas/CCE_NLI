import nltk
from nltk.corpus import stopwords

path="snli_1.0_dev.tok"
def download_stopwords():
    nltk.download('stopwords')
    stopwords.words('english')
    stop_words = set(stopwords.words('english'))
    return stop_words

def readFile(path):
    lines=[]
    with open(path, 'r') as f:
        lines=f.readlines()
    f.close()  
    return lines

def get_words(lines):
    words=[]
    for sent in lines:
        words.extend(sent.split(' '))
    return words

def remove_stop_words(words, stop_words):

    filtered_sentence = [word.lower() for word in words if not word.lower() in stop_words]
    return filtered_sentence

def remove_duplicates(filtered_sentence):
    return set(filtered_sentence)

stop_words= download_stopwords()
lines = readFile(path)
words = get_words(lines)
filtered_sentence = remove_stop_words(words, stop_words)
unique_words= remove_duplicates(filtered_sentence)
print(len(unique_words))