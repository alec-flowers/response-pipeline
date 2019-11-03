# Disaster Response Pipelines
##Overview
When disasters strike the influx of data from people who need help can be overwhelming. One big problem is how to connect the people who need help with the correct emergency service. A potential solution is to use machine learning to take messages from social media, classify them, and then re-direct them to the correct agency depending on the contents of the message. This would allow for faster response and more effective resource allocation to those people who need that service most. 

##ETL Pipeline
I was given real life message data sent from disasters that had already been classified into 35 categories. To get the data into the correct format for a ML model to run I built an Extract - Transform - Load pipeline. In this step I merged the message and classification data, turned the classification data into columns of binary values, and stored the result in a SQLite database. 

Categories: 'related', 'request', 'offer', 'aid_related', 'medical_help',
       'medical_products', 'search_and_rescue', 'security', 'military',
       'water', 'food', 'shelter', 'clothing', 'money', 'missing_people',
       'refugees', 'death', 'other_aid', 'infrastructure_related', 'transport',
       'buildings', 'electricity', 'tools', 'hospitals', 'shops',
       'aid_centers', 'other_infrastructure', 'weather_related', 'floods',
       'storm', 'fire', 'earthquake', 'cold', 'other_weather',
       'direct_report'

##ML Pipeline
Next I buld a machine learning pipeline that would be able to turn the message data into something a computer could read and analyze. This involved turning the message data into lemmatized tokens, vectorizing these tokens and finally using Term Frequency - Inverse Term Frequency which takes the vectorized data and statisitcally determines how important a term is to the document. 

In order to optimize the ML algorithm I used GridSearch and input a number of different hyperparameters to run and selected the best ones. I saved the output of this model for later use. 


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

##Web App
The final part was to deploy a web app that would allow someone to type in a message and our trained alogrithm would classify it into the proper categories. 

I built two additional visualizations of our data that
1) shows how frequent certain categories were in the dataset
2) displays the count of messages bucketed by the number of categories 


##Instructions to Run:
In the projects root directory run these commands
1) ETL Pipeline - Clean data and store in SQLite
    python data/process_data.py messages.csv categories.csv DisasterResponse.db

2) ML pipeline - train random forest classifier and saves
    python models/train_classifier.py DisasterResponse.db classifier.pkl

3) Web App - Deploy web app
    cd into the app folder
    python run.py

4) Go to http://0.0.0.0:3001/


##File Structure
Here is the file structure of the project: 

- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py #etl pipeline
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py #ml pipeline
|- model.pkl  # saved model 

- packages #
|- __init__.py
|- utils.py

- setup.py
- README.md