import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from packages.utils import get_first_verb, get_char_num

app = Flask(__name__)

def tokenize(text):
    """
        Split text and lemmatize text. 
        
        Input:
            text

        Output:
            clean_tokens (lst) : Cleaned text
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterDatabase.db')
df = pd.read_sql_table('cleaned_disaster_data', engine)

# load model
model = joblib.load("../models/model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_counts = list(df.iloc[:,4:].sum().values)
    category_names = list(df.iloc[:,4:].sum().index)
    
    number_of_classifications = list(df.iloc[:,4:].sum(axis=1).value_counts().index)
    number_of_messages = list(df.iloc[:,4:].sum(axis=1).value_counts().values)


    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts,
                    marker =  {
                        'color' : category_counts,
                        'colorscale' : 'Viridis'
                    }
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'categoryorder' : 'array',
                    'categoryarray' : [x for _, x in sorted(zip(category_counts,category_names))]
                }
            }
        },
        #
        {
            'data': [
                Bar(
                    x=number_of_classifications,
                    y=number_of_messages,
                    marker = {
                        'color' : number_of_messages,
                        'colorscale':'magma'
                    }
                )
            ],

            'layout': {
                'title': 'Distribution of number of message classifications',
                'yaxis': {
                    'title': "Number of Messages"
                },
                'xaxis': {
                    'title': "Number of Classifications"
                }
            }
        },
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()