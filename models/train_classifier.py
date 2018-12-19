import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, MetaData, Table
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

nltk.download(['punkt','stopwords', 'wordnet'])
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#From Udacity training material
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'




def count_true_false(the_actual, the_predicted):
    """
    Function returns the numbers for True Positive, True Negative, False Positive and False Negative,
    calculated based on arrays of actual and predicted values
    
    Input:  the_actual: array. An array of actualt numbers 
            the_predicted: array. An array of predicted numbers
       
    Output: True Positive: number. Sum of True Positive
            True Negative: number. Sum of True Negative
            False Positive: number. Sum of False Positive
            False Negative: number. Sum of False Negative
    
    """
    truepostive = 0
    truenegative = 0
    falsepositive=0
    falsenegative=0

    for actual,pred in zip(the_actual, the_predicted):
        if actual == pred:
            if actual == True:
                truepostive+=1
            else:
                truenegative+=1
        else:
            if actual == False:
                falsepositive+=1
            else:
                falsenegative+=1
    return truepostive,truenegative,falsepositive,falsenegative



def get_eval_result(actual_values, predicted_values, columns):  
    """
    Function returns a dataframe with evaluation result after comparing actual and predicted binary values
    
    Input:    actual_values: array. an array holding actual binary values.
              predicted_values: array. an array holding predicted binary values
              columns: list of strings, text describing the predicted fields
       
    Output:   eval_result_df: dataframe containing the calculated accuracy, precision, 
              recall, f1  as well as values for True Positive, True Negative, False Positive and False Negative
              
    Function based on idea from 
    REF: https://github.com/gkhayes/disaster_response_app/blob/master/ML%20Pipeline%20Preparation.ipynb
    """
    
    eval_result_df = pd.DataFrame(columns=["accuracy","precision","recall","F1","TP","TN","FP","FN"])
    
    tot_TP, tot_TN, tot_FP, tot_FN = 0,0,0,0
    
    actual_values_total = np.empty([0,0])
    predicted_values_total = np.empty([0,0])
    
    
    colno = 0
    for column in columns:
        accuracy = accuracy_score(actual_values[:, colno], predicted_values[:, colno])
        precision = precision_score(actual_values[:, colno], predicted_values[:, colno])
        recall = recall_score(actual_values[:, colno], predicted_values[:, colno])       
        f1 = f1_score(actual_values[:, colno], predicted_values[:, colno])
        TP,TN,FP,FN = count_true_false(actual_values[:, colno], predicted_values[:, colno])
        eval_result_df.loc[column] = [accuracy,precision,recall,f1,TP,TN,FP,FN]
        tot_TP += TP
        tot_TN += TN
        tot_FP += FP
        tot_FN += FN
        colno += 1
    
    accuracy = ((tot_TP+tot_TN)/(tot_TP+tot_TN+tot_FP+tot_FN))
    #print(f"accuracy:{accuracy}")    
    precision = (tot_TP)/(tot_TP+tot_FP)
    #print(f"precision:{precision}")    
    recall = (tot_TP)/(tot_TP+tot_FN)
    #print(f"recall:{recall}")    
    F1 = 2*(precision*recall)/(precision+recall)
    #print(f"F1:{F1}")    
    
    eval_result_df.loc["*** total_accross_all_features ***"] = [ accuracy, precision, recall, f1, "", "", "", "" ]
        
    return eval_result_df

    


def load_data(database_filepath):
    """
    Function loads data from a known table in a database specified by input parameter and returns the 
    data split into X,y and column values
    
    Input:  database_filepath: string. The file name of the database
            
    Output: X: DataFrame, holding one column of text message
            y: DataFrame, holding columns with text features indicating the categories/topics the message
                is related to
            y.columns: index. An index of all the DataFrame columns acting as categories/topics
              
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table("DisasterResponseMessages",engine)
    X = df["message"]
    y = df.drop(["id","message","original","genre"], axis='columns')
    return X, y, y.columns 
    #pass


def tokenize(text):
    """
    Function is returning a tokenized representation of an input string. As part of the processing, 
    URL's are removed, as well as short words and stop words. Lemmatizing is also done after tokenization
    
    Input:  text: string. A message to be tokenized
            
    Output: clean_tokens: list of words having been tokenized and lemmatized
              
    Based on Udacity training material
    """
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "")

    tokens = word_tokenize(text)
    # removing short words
    tokens = [token for token in tokens if len(token) > 2]
    
    # removing stopwords
    tokens = [token for token in tokens if token not in list(set(stopwords.words('english')))]
        
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok)#.lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens    
    #pass


def build_model():
    """
    Function returns an ML pipeline model, running GridSearch using a RandomForesClassifier
    Input:   NA
    Output:  cv: GridSearchCV. a model prepared for GridSearch, already set with some parameters
              
    """
    pipeline = Pipeline([
        #('vect', CountVectorizer(tokenizer=tokenize,max_features=None,max_df=1)),
        #('tfidf', TfidfTransformer(use_idf=True)),
        #('clf', MultiOutputClassifier(MultinomialNB(alpha=1.0), n_jobs=1))
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {'vect__min_df': [1, 5],
              #'tfidf__use_idf':[True, False],
              #'clf__estimator__n_estimators':[10, 25], 
              #'clf__estimator__min_samples_split':[2, 5, 10]
                 }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose = 2)

    return cv
    #pass


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function evaluates a given model, based on input data and returns a dataframe showing the results
    per feature as well as a summary line accross all features
    Input    model: sklearn.model. A fitted model ready to use for prediction  
    Output   result_df: pd.DataFrame. A Pandas Dataframe holding the model score 
    """
    Y_pred = model.predict(X_test)
    result_df = get_eval_result(np.array(Y_test), Y_pred, category_names)
    print(result_df);


def save_model(model, model_filepath):
    """
    Function saves a given model as a pickle-file
    Input    model: sklearn.model. A fitted model ready to use for prediction  
    Output   NA
    """
    
    joblib.dump(model, model_filepath)
    #pass 


def main():
    """
    Function takes a set of onput parameters indicating data source and where to store a 
    trained model as a pickle file 

    Input   database_filepath: sys.argv, text indicating the location of a database file
            model_filepath:  sys.argv, text indicating where to save the model pickle file
    Output  NA
    
    Function pattern is given by Udacity
    """
    np.random.seed(42)
    #break
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        print(f"model.best_params: {model.best_params_}")
        
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