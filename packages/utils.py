from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')


def tokenize(text):
    """
    Tokenize text, take out stop words, lemmatize text 

    Input:
    text (str)

    Output:
    lemmed (list) : List of strings 
    """

    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = word_tokenize(text)
    no_stop_words = [w for w in words if w not in stopwords.words("english")]
    lemmed_word_list = [WordNetLemmatizer().lemmatize(w).lower().strip() for w in no_stop_words]
    return lemmed_word_list

class get_first_verb(BaseEstimator,TransformerMixin):
    def starting_verb(self,text):
        """
        Return true if first word is a verb, otherwise return false. 
        """
        sent_list = sent_tokenize(text)
        
        for sentence in sent_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            try:
                first_word, first_tag = pos_tags[0]
            except:
                return False
                continue
            if first_tag in ['VB', 'VBP']:
                return True
        return False

    def fit(self, X, y=None):
        """
        Function from baseclass. Fits data
        """
        return self

    def transform(self, X):
        """
        Function from baseclass. Transforms data
        """
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)    

class get_char_num(BaseEstimator,TransformerMixin):
    def count_char(self,text):
        """
        Counts number of characters
        """
        return len(text)
    def fit(self,X,y=None):
        """
        Function from baseclass. Fits data
        """
        return self
    def transform(self,X):
        """
        Function from baseclass. Transforms data
        """
        X_count = pd.Series(X).apply(self.count_char)
        return pd.DataFrame(X_count)
