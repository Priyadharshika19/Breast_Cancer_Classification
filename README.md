# Breast_Cancer_Classification
  Dataset - Breast Cancer
  ML - Supervised learning (Classification)
  
  1. **import the packages/libraries**
      import pandas as pd
      import numpy as np
      import seaborn as sns
      import matplotlib.pyplot as plt
      from sklearn.model_selection import train_test_split, cross_val_score
      from sklearn.preprocessing import StandardScaler
      from sklearn.linear_model import LogisticRegression
      from sklearn.tree import DecisionTreeClassifier
      from sklearn.neighbors import KNeighborsClassifier
      from sklearn.ensemble import RandomForestClassifier
      from sklearn.pipeline import make_pipeline
      from sklearn.svm import SVC
      import xgboost as xgb
      from sklearn.naive_bayes import GaussianNB
      from sklearn.metrics import accuracy_score, roc_auc_score,f1_score
      from mlxtend.plotting import plot_decision_regions
     
  2. **Load data**

      temp=pd.read_csv("/content/drive/MyDrive/dataset/cancer.csv")  #load dataset
      df=temp.copy()  #copy of dataset
      pd.set_option('display.max_columns', None) #display all values
      print(df.head())  # view first 5 rows
      print(df.shape)  #shape of dataset
      (569, 33)
     
     
  4. View and delete data
  







# Breast_Cancer_Classification
**Step 1:**
**import the packages/libraries**
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import SVC
    import xgboost as xgb
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score, roc_auc_score,f1_score
    from mlxtend.plotting import plot_decision_regions
**Step 2:**
**load the data**
    temp=pd.read_csv("/content/drive/MyDrive/dataset/cancer.csv")  #load dataset
    df=temp.copy()
    pd.set_option('display.max_columns', None)
    print(df.head())
    print(df.shape)
