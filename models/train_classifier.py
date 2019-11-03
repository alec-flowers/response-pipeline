import sys

from sqlalchemy import create_engine

import re
import numpy as np
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from sklearn.metrics import confusion_matrix, classification_report,roc_curve, hamming_loss, accuracy_score, make_scorer, recall_score,precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.externals import joblib
from packages.utils import get_first_verb, get_char_num

def load_data(database_filepath):
    """
    Load cleaned data from SQL Lite database and break it up into X (data) and y (labels).
    
    Input:
        database_filepath (str) : name you want to call database

    Output:
        X (list) : Message data in string format
        y (dataframe) : Classification labels for the messagew data
        y.columns (list) : List of column names
    """
    engine = create_engine('sqlite:///data/{}'.format(database_filepath))
    conn = engine.connect()
    df = pd.read_sql_query(""" Select * FROM cleaned_disaster_data""",conn)
    X = df.message.values
    y = df.drop(['id','message','original','genre'],axis=1)
    return X,y,y.columns

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

def build_model():
    """
    Define pipeline to 
        1) transform sentences into features that we can train model with.
        2) Create Random Forest Model to train 
    Set grid search parameters and scorers to optimize model.

    Output:
    cv (GridSearchCV object) : model ready to be trained
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', get_first_verb()),
            ('text_length',get_char_num())
        ])),

        ('clf', RandomForestClassifier())
    ])
    parameters = {
        'clf__min_samples_split': [3, 5, 10], 
        'clf__n_estimators' : [100, 300],
        'clf__max_depth': [3, 5, 15, 25],
        'clf__max_features': [3, 5, 10, 20]
        }

    scorers = {
        'precision_score': make_scorer(precision_score,average = 'micro'),
        'recall_score': make_scorer(recall_score,average = 'micro'),
        'accuracy_score': make_scorer(accuracy_score)
        }

    cv = GridSearchCV(pipeline,parameters,scoring = scorers,refit = 'recall_score')

    return cv

def evaluate_model(model,X_test,y_true,category_names):
    """
    Predict test data labels. Print out various model evaluation metrics
    comparing test and actual labels. 

    Input:
        model (GridSearchCV): 
        X_test : test data model has never seen before
        y_test : test labels to compare to truth
        category_names (list): Names of classification columns to label data

    Output:
        None : Print out various metrics
    """
    y_pred = pd.DataFrame(model.predict(X_test),columns = category_names)
    y_true = pd.DataFrame(y_true,columns = category_names)

    print('Classification Report')
    print(classification_report(y_true,y_pred,target_names = category_names))
    print('Model Accuracy: {}'.format(accuracy_score(y_true,y_pred)))
    print('Model Hamming Loss: {}'.format(hamming_loss(y_true,y_pred)))
    
    #Print out classification report for every class we are trying to predict
    for name in category_names:
        print(name)
        print(classification_report(y_true[name],y_pred[name]))

def save_model(model, model_filepath):
    """
    Use joblib to save snapshot of trained model
    """
    joblib.dump(model, model_filepath)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()