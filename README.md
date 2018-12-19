# Disaster Response Pipeline Project - Udacity student project

This is a ML project at Udacity, aiming to create an app to predict correct category of messages received during disasters. The workspace IDE and datafiles, as well as structure
of project files are specified by Udacity

# Project outline from project description
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md


# App description
The app created is a web based solution based on ML and Python, aiming to categorize messages
It also shows some graphs describing the data used as input for the project

# Final project files:
- app
| - template
| |- master.html  # main page of web app (original from Udacity)
| |- go.html  # classification result page of web app (original from Udacity)
|- run.py  # Flask file that runs app (edited to add visuals)

- data
|- disaster_categories.csv  # data to process (original from Udacity)
|- disaster_messages.csv  # data to process (original from Udacity)
|- process_data.py (modified to process data) 
|- process_data.py.original (original from Udacity)
|- DisasterResponse.db   # database to save clean data to (generated by process_data.py)
|- workspace_utils.py  #File provided as part of training material at Udacity for other project. Used to instruct workspace to extend session for long running jobs

- models
|- train_classifier.py (modified to train data)
|- train_classifier.py.original  (original from Udacity)
|- classifier.pkl  # saved model (generated by train_classifier.py, zipped due to size)


### Instructions from Udacity on how to run the app:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

4. From project description:
    - Running the Web App from the Project Workspace IDE
   When working in the Project Workspace IDE, here is how to see your Flask app.
   Open a new terminal window. You should already be in the workspace folder, but if not, then
   use terminal commands to navigate inside the folder with the run.py file.
   
   Type in the command line:  
   python run.py
   
   Your web app should now be running if there were no errors.
   Now, open another Terminal Window.
   Type:
   env|grep WORK

	You'll see output that looks something like this:
		root@a698ff32b38e:/home/workspace# env|grep WORK
		WORKSPACEDOMAIN=udacity-student-workspaces.com
		WORKSPACEID=view6914b2f4
		root@a698ff32b38e:/home/workspace#

	In a new web browser window, type in the following:
    https://SPACEID-3001.SPACEDOMAIN
	In this example, that would be: "https://view6914b2f4-3001.udacity-student-workspaces.com/" (Don't follow this link now, this is just an example.)
	Your SPACEID might be different.

# Comments
The data have been run through several ML algorithms using GridSearch. The overall precision is best when using MultinomialNB as classifier, but too many categories have not a single TruePositive identification, and therefore I ended up with RandomForest which gets some hits in more categories, but has a lower precision. It concerns that a situation where there is a message indicating "Fire" is difficult to identify.

Resons why precision is so low might be overfitting.
A suggestion for improvement could be to investigate "feature_importances" in RandomFores to see if there are special words that should be excluded or handled in a special way.
