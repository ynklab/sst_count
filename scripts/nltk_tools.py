from nltk.tokenize import RegexpTokenizer  # for nltk word tokenization
from nltk.stem.wordnet import WordNetLemmatizer

lmtzr = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r"(?x)(?:[A-Za-z]\.)+| \w+(?:\w+)*")
emb_size = 300

stop_words = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she','her', 'hers', 'herself',
    'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
    'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
    'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're',
    've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven',
    'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won',
    'wouldn']
    
nums = [str(i) for i in range(2100)] + ["one", "two", "three", "four", "five", 
    "six", "seven", "eight", "nine", "ten"] + ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09"]

def lemmatization(word):
    return lmtzr.lemmatize(word)
    
# tokenization
def tokenization(sentence, stop_word_flag):
    words = tokenizer.tokenize(sentence.lower())
    if stop_word_flag == 1:
        words = [lemmatization(w1) for w1 in words]
        words = [w for w in words if w not in stop_words]
    if stop_word_flag == 2:
        words = [lemmatization(w1) for w1 in words]
        new_words = []
        for i in range(len(words)):
            if words[i] in stop_words + nums: 
                if i + 1 < len(words) and words[i] == "won" and words[i+1] != "t": new_words.append("win")
            else:
                new_words.append(words[i])
        words = new_words
    return words

if __name__ == '__main__':
    print(tokenization("He won 2 awards.", 2))