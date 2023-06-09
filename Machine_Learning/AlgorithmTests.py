import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, fbeta_score, make_scorer
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
import multiprocessing as mp
import pickle

# Creating a function that uses category_id to determine if a cage is present.
def cage_detection(row):
    if (row['category_id'] == 0) or (row['category_id'] == 1):
        return 1
    else:
        return 0


def GridSearchTestsSVCweighted(params, scorer, path, randomState=42):
    # Defining the different columns and features
    columns = ['video_id', 'path', 'width', 'height', 'fps', 'total_frames', 'file_name', 'category_id',
               'category_name', 'supercategory', 'color', 'metadata', 'pass_id', 'Frame', 'BoudingBox', 'Area',
               'Circularity', 'Convexity', 'Rectangularity', 'Elongation', 'Eccentricity', 'Solidity']
    features = ['Area', 'Circularity', 'Convexity', 'Rectangularity', 'Elongation', 'Eccentricity', 'Solidity']

    # Import csv with all data
    df = pd.read_csv('Extracted_Features/featuresExtractedSorted_noDoubleFrames.csv', usecols=columns)

    # Make a new cage column that applies the function to each row
    df['Cage'] = df.apply(lambda row: cage_detection(row), axis=1)

    # Make a dataframe containing all features
    X = df[features]

    # Make a column with results of the classification
    y = df['Cage']

    # Prepare test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=randomState)
    scaler = StandardScaler().fit(X_train)

    # Scale test data
    print(X_train[:5])
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(X_train_scaled[:5])
    # Do a cross validation using GridSearchCV
    optimal_params = GridSearchCV(
        SVC(),
        params,
        cv=5,
        scoring=scorer,
        refit=True,
        verbose=1,
        n_jobs=-1
    )
    
    # Fit the parameters using GridSearchCV
    optimal_params.fit(X_train_scaled, y_train)
    
    # Print the optimal parameters
    print(optimal_params.best_params_)

    # Record the results of the grid search
    Results = pd.DataFrame(optimal_params.cv_results_)
    Results.to_csv(f'{path}GridSearchResults.csv')

    # Uses the best estimator from grid search to predict test data
    clf = optimal_params.best_estimator_
    clf.fit(X_train_scaled, y_train)
    predictions = clf.predict(X_test_scaled)
    cm = confusion_matrix(y_test, predictions, labels=clf.classes_)

    # Display new confusion matrix
    after = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Is not a cage", "Is a cage"])
    after.plot()
    plt.title(f'F4 Score = {fbeta_score(y_test, predictions, beta=4)}')
    plt.savefig(f'{path}after_CrossValidation.pdf')
    plt.savefig(f'{path}after_CrossValidation.png')
    plt.clf()

    # Groups by passing, and then computes confusion matrix of that
    pass_test = y_test.copy()
    frame_id = pd.DataFrame([str(df["video_id"][i]) + "-" + str(df["pass_id"][i]) for i in range(len(df))],
                            columns=["frame_id"])
    y_cat_df = pd.concat([pass_test, frame_id], axis=1).dropna()
    y_grouped_df = y_cat_df.groupby(by=["frame_id"]).sum()
    y_test_pass = y_grouped_df.divide(y_grouped_df).fillna(0)

    pass_pred = pd.DataFrame(np.array([pass_test.index, predictions]).transpose(), columns=["index", "cage"]).set_index(
        "index")
    pred_cat_df = pd.concat([pass_pred, frame_id], axis=1).dropna()
    pred_grouped_df = pred_cat_df.groupby(by=["frame_id"]).sum()
    pred_test_pass = pred_grouped_df.divide(pred_grouped_df).fillna(0)

    cm = confusion_matrix(y_test_pass, pred_test_pass, labels=clf.classes_)
    passCM = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Is not a cage", "Is a cage"])
    passCM.plot()
    plt.title(f'F4 Score = {fbeta_score(y_test_pass, pred_test_pass, beta=4)}')
    plt.savefig(f'{path}pass_CrossValidation.pdf')
    plt.savefig(f'{path}pass_CrossValidation.png')
    plt.clf()

    # Save the model
    filename = f'{path}finalized_model.sav'
    pickle.dump(clf, open(filename, 'wb'))
    # Save the model
    filename = f'{path}finalized_scaler.sav'
    pickle.dump(scaler, open(filename, 'wb'))
    plt.close()

    
    del df
    return locals()


def GridSearchTestsSVC(params, scorer, path, randomState=42):
    # Defining the different columns and features
    columns = ['video_id', 'path', 'width', 'height', 'fps', 'total_frames', 'file_name', 'category_id',
               'category_name', 'supercategory', 'color', 'metadata', 'pass_id', 'Frame', 'BoudingBox', 'Area',
               'Circularity', 'Convexity', 'Rectangularity', 'Elongation', 'Eccentricity', 'Solidity']
    features = ['Area', 'Circularity', 'Convexity', 'Rectangularity', 'Elongation', 'Eccentricity', 'Solidity']

    # Import csv with all data
    df = pd.read_csv('Extracted_Features/featuresExtractedSorted_noDoubleFrames.csv', usecols=columns)

    # Make a new cage column that applies the function to each row
    df['Cage'] = df.apply(lambda row: cage_detection(row), axis=1)

    # Make a dataframe containing all features
    X = df[features]

    # Make a column with results of the classification
    y = df['Cage']

    # Prepare test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=randomState)
    scaler = StandardScaler().fit(X_train)

    # Scale test data
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Do a cross validation using GridSearchCV
    optimal_params = GridSearchCV(
        SVC(),
        params,
        cv=5,
        scoring=scorer,
        refit=True,
        verbose=1,
        n_jobs=-1
    )

    # Fit the parameters using GridSearchCV
    optimal_params.fit(X_train_scaled, y_train)

    # Print the optimal parameters
    print(optimal_params.best_params_)

    # Record the results of the grid search
    Results = pd.DataFrame(optimal_params.cv_results_)
    Results.to_csv(f'{path}GridSearchResults.csv')

    # Uses the best estimator from grid search to predict test data
    clf = optimal_params.best_estimator_
    clf.fit(X_train_scaled, y_train)
    predictions = clf.predict(X_test_scaled)
    cm = confusion_matrix(y_test, predictions, labels=clf.classes_)

    # Display new confusion matrix
    after = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Is not a cage", "Is a cage"])
    after.plot()
    plt.title(f'F4 Score = {fbeta_score(y_test, predictions, beta=4)}')
    plt.savefig(f'{path}after_CrossValidation.pdf')
    plt.savefig(f'{path}after_CrossValidation.png')
    plt.clf()

    # Groups by passing, and then computes confusion matrix of that
    pass_test = y_test.copy()
    frame_id = pd.DataFrame([str(df["video_id"][i]) + "-" + str(df["pass_id"][i]) for i in range(len(df))],
                            columns=["frame_id"])
    y_cat_df = pd.concat([pass_test, frame_id], axis=1).dropna()
    y_grouped_df = y_cat_df.groupby(by=["frame_id"]).sum()
    y_test_pass = y_grouped_df.divide(y_grouped_df).fillna(0)

    pass_pred = pd.DataFrame(np.array([pass_test.index, predictions]).transpose(), columns=["index", "cage"]).set_index(
        "index")
    pred_cat_df = pd.concat([pass_pred, frame_id], axis=1).dropna()
    pred_grouped_df = pred_cat_df.groupby(by=["frame_id"]).sum()
    pred_test_pass = pred_grouped_df.divide(pred_grouped_df).fillna(0)

    cm = confusion_matrix(y_test_pass, pred_test_pass, labels=clf.classes_)
    passCM = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Is not a cage", "Is a cage"])
    passCM.plot()
    plt.title(f'F4 Score = {fbeta_score(y_test_pass, pred_test_pass, beta=4)}')
    plt.savefig(f'{path}pass_CrossValidation.pdf')
    plt.savefig(f'{path}pass_CrossValidation.png')
    # Save the model
    filename = f'{path}finalized_model.sav'
    pickle.dump(clf, open(filename, 'wb'))
    # Save the model
    filename = f'{path}finalized_scaler.sav'
    pickle.dump(scaler, open(filename, 'wb'))
    plt.close()
    del df
    return locals()

def GridSearchTestsKNN(params, scorer, path, randomState=42, mahalanobis=False):
    # Defining the different columns and features
    columns = ['video_id', 'path', 'width', 'height', 'fps', 'total_frames', 'file_name', 'category_id',
               'category_name', 'supercategory', 'color', 'metadata', 'pass_id', 'Frame', 'BoudingBox', 'Area',
               'Circularity', 'Convexity', 'Rectangularity', 'Elongation', 'Eccentricity', 'Solidity']
    features = ['Area', 'Circularity', 'Convexity', 'Rectangularity', 'Elongation', 'Eccentricity', 'Solidity']

    # Import csv with all data
    df = pd.read_csv('Extracted_Features/featuresExtractedSorted_noDoubleFrames.csv', usecols=columns)

    # Make a new cage column that applies the function to each row
    df['Cage'] = df.apply(lambda row: cage_detection(row), axis=1)

    # Make a dataframe containing all features
    X = df[features]

    # Make a column with results of the classification
    y = df['Cage']

    # Prepare test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=randomState)
    scaler = StandardScaler().fit(X_train)
    
    # Scale test data
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if(mahalanobis):
        params["metric"] = ["mahalanobis"]
        params["metric_params"] = [{"V": np.cov(X_train_scaled, rowvar=False)}]

    # Do a cross validation using GridSearchCV
    optimal_params = GridSearchCV(
        KNeighborsClassifier(),
        params,
        cv=5,
        scoring=scorer,
        refit=True,
        verbose=1,
        n_jobs=-1,
    )

    # Fit the parameters using GridSearchCV
    optimal_params.fit(X_train_scaled, y_train)

    # Print the optimal parameters
    print(optimal_params.best_params_)

    # Record the results of the grid search
    Results = pd.DataFrame(optimal_params.cv_results_)
    Results.to_csv(f'{path}GridSearchResults.csv')

    # Uses the best estimator from grid search to predict test data
    clf = optimal_params.best_estimator_
    clf.fit(X_train_scaled, y_train)
    predictions = clf.predict(X_test_scaled)
    cm = confusion_matrix(y_test, predictions, labels=clf.classes_)

    # Display new confusion matrix
    after = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Is not a cage", "Is a cage"])
    after.plot()
    plt.title(f'F4 Score = {fbeta_score(y_test, predictions, beta=4)}')
    plt.savefig(f'{path}after_CrossValidation.pdf')
    plt.savefig(f'{path}after_CrossValidation.png')
    plt.close()

    # Groups by passing, and then computes confusion matrix of that
    pass_test = y_test.copy()
    frame_id = pd.DataFrame([str(df["video_id"][i]) + "-" + str(df["pass_id"][i]) for i in range(len(df))],
                            columns=["frame_id"])
    y_cat_df = pd.concat([pass_test, frame_id], axis=1).dropna()
    y_grouped_df = y_cat_df.groupby(by=["frame_id"]).sum()
    y_test_pass = y_grouped_df.divide(y_grouped_df).fillna(0)

    pass_pred = pd.DataFrame(np.array([pass_test.index, predictions]).transpose(), columns=["index", "cage"]).set_index(
        "index")
    pred_cat_df = pd.concat([pass_pred, frame_id], axis=1).dropna()
    pred_grouped_df = pred_cat_df.groupby(by=["frame_id"]).sum()
    pred_test_pass = pred_grouped_df.divide(pred_grouped_df).fillna(0)

    cm = confusion_matrix(y_test_pass, pred_test_pass, labels=clf.classes_)
    passCM = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Is not a cage", "Is a cage"])
    passCM.plot()
    plt.title(f'F4 Score = {fbeta_score(y_test_pass, pred_test_pass, beta=4)}')
    plt.savefig(f'{path}pass_CrossValidation.pdf')
    plt.savefig(f'{path}pass_CrossValidation.png')
    plt.close()
    # Save the model
    filename = f'{path}finalized_model.sav'
    pickle.dump(clf, open(filename, 'wb'))
    # Save the model
    filename = f'{path}finalized_scaler.sav'
    pickle.dump(scaler, open(filename, 'wb'))
    del df
    return locals()

if __name__ == "__main__":
    # create a process for each video
    processes = []

    functions = [GridSearchTestsSVCweighted, GridSearchTestsSVC, GridSearchTestsKNN, GridSearchTestsKNN]
    arguments = [
                    {'params':{'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'linear'],
                     'class_weight': [None, "balanced", {0: 1, 1: 16}, {0: 1, 1: 4}, {0: 1, 1: 3}], "random_state":[42]},
                    'scorer':make_scorer(fbeta_score, beta=4),'path':"Test_Results/SVM_Weights/"}, 
                    {'params':{'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'linear'],
                    'class_weight': [None], "random_state": [42]}, 'scorer':make_scorer(fbeta_score, beta=4),'path':"Test_Results/SVM_No_Weights/"},
                    {'params':{"n_neighbors":[1,3,5,7,9], "weights":["uniform", "distance"], "metric":["euclidean"]},
                    'scorer':make_scorer(fbeta_score, beta=4), 'path':"Test_Results/KNN_Euclidean/"},
                    {'params':{"n_neighbors": [1, 3, 5, 7, 9], "weights": ["uniform", "distance"]},
                    'scorer':make_scorer(fbeta_score, beta=4), 'path':"Test_Results/KNN_Mahalanobis/", 'mahalanobis':True}
                    ]

    for i, argument in enumerate(arguments):
        p = mp.Process(target=functions[i],  kwargs=(argument))  
        processes.append(p)
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()