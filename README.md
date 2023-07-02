# Breast_Cancer_Classification

  Dataset - Breast Cancer
  ML - Supervised learning (Classification)
  
  1. **import the packages/libraries**

    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.model_selection import rain_test_split, cross_val_score
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
   
  3. **Checking Dataset (Balanced or Imbalanced)**
     
    val_cnt=df['diagnosis'].value_counts()
    print("values count: ")
    print(val_cnt)
    print("% of B: ",val_cnt["B"]/len(df['diagnosis'])*100)
    print("% of M: ",val_cnt["M"]/len(df['diagnosis'])*100)

![cancer1](https://github.com/Priyadharshika19/Breast_Cancer_Classification/assets/129640468/ad0805f1-528d-420d-b91f-7e4b2b1d7bbb)


    Here dataset is balanced.
    if class B or class M > 70%  then It is umbalanced and the model will be biased 
  
  4. **Relationship between features**
     
    import seaborn as sns
    plt.figure(figsize=(30,30))
    sns.heatmap(df.corr(),annot=True,fmt=".0%")
    plt.show()

 ![c3](https://github.com/Priyadharshika19/Breast_Cancer_Classification/assets/129640468/1fbf4288-ca6f-4094-bd4d-3190712ea4a2)


  5. **Feature Engineering**
     
    1. missing or null values
        sns.heatmap(df.isna())
        df.isna().sum()
     
![c2](https://github.com/Priyadharshika19/Breast_Cancer_Classification/assets/129640468/4d9b9a20-de44-4857-b73f-ca51907be2ad)![c2 jpg](https://github.com/Priyadharshika19/Breast_Cancer_Classification/assets/129640468/cb538fea-94b1-46f2-a0c6-4d43c3f17550)

        All values in Unknown 32 are Null values so drop column Unknown 32
        df.dropna(axis=1,inplace=True)
     
    2. unrequired features
    
        del df["id"]   
        
    3. Duplicates
    
        df.duplicated().sum()
        In this dataset, there is no duplicates.
        if there is any, drop duplicates
     
    4. Incorrect format
    
        df.info()
        
![c4](https://github.com/Priyadharshika19/Breast_Cancer_Classification/assets/129640468/b0757ed3-f1fb-4fe6-84e8-fb565afc4c03)

        There is no incorrect format

    5. Outliers
    
      1.Create DataFrame with column names, minimum value,lower threshold value, maximum value, maximum threshold value

        df.describe()
        def iqr_val(col):
          #print(type(col))
          iqr=df[col].quantile(0.75)-df[col].quantile(0.25)
          up_Threshold=df[col].quantile(0.75)+(1.5*iqr)
          low_Threshold=df[col].quantile(0.25)-(1.5*iqr)
          return up_Threshold, low_Threshold

        outliers_dict={"columns":[],"min_val":[],"low_Threshold":[],"max_val":[],"up_Threshold":[]}
        for i in df.columns[1:] :
          up_Threshold,low_Threshold=iqr_val(i)
          outliers_dict["columns"].append(i)
          outliers_dict["min_val"].append(df[i].min())
          outliers_dict["low_Threshold"].append(low_Threshold)
          outliers_dict["max_val"].append(df[i].max())
          outliers_dict["up_Threshold"].append(up_Threshold)
          #df[i]=df[i].clip(low_Threshold,up_Threshold)
        outliers_df=pd.DataFrame(outliers_dict)
        #df["area_worst"].max()
        outliers_df
        
![c5](https://github.com/Priyadharshika19/Breast_Cancer_Classification/assets/129640468/b021e7d7-78ac-4b75-b547-a90fc8e2188b)

        a=df["area_worst"]
        count=0
        for i in a:
          if i > 1937.050000:
            #print(i)
            count=count+1
        print("outliers count: ",count)
        print("outliers : ",round(count/len(df["area_worst"])*100),"%") 
        
  ![c6](https://github.com/Priyadharshika19/Breast_Cancer_Classification/assets/129640468/92094fd1-b997-4c1e-b278-8e4600834d89)

      2.**Replace outliers with lower and upper threshold values **
        for col in df.columns[1:]:
            for ele in range(0,len(df[col])):
              for j in outliers_df.loc[outliers_df["columns"]==col, "low_Threshold"]:
                  if df[col][ele] < j:
                      df[col][ele] = j
              for i in outliers_df.loc[outliers_df["columns"]==col, "up_Threshold"]:
                  if df[col][ele] > i:
                      df[col][ele] = i

  6. **Encode the categorical data**

    df["diagnosis"].replace(["M","B"],[1,0],inplace=True)
    df["diagnosis"].unique()
        array([1, 0])

  7. **EDA**
     
    sns.pairplot(df, hue="diagnosis")
    Sample graph:
    
![c9](https://github.com/Priyadharshika19/Breast_Cancer_Classification/assets/129640468/37e9b0d0-3daf-4eaf-b6b2-5d9649c7459c)

  9. **Split data into Train and Test**

    Splitting data in ratio 80:20

      X=df.loc[:,df.columns[1:]].values
      y=df.loc[:,"diagnosis"].values
      X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
      len(X_train),len(X_test)

            (455, 114)
            
  9. **Scale data**

    scaling is not mandatory for all models

      scaler=StandardScaler()
      scaler.fit(X_train,y_train)
      scaled_x_train=scaler.transform(X_train)
      scaled_x_test=scaler.transform(X_test)
      
  11. **Model fitting, predicting target and finding accuracy_score**
    LogisticRegression       
    KNN       
    Decision_tree       
    Random forest
    SVC       
    XGBoost            
    GaussianNB       

    def clf(df,clf):

      X=df.loc[:,df.columns[1:]].values
      y=df.loc[:,"diagnosis"].values
      X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
      scaler=StandardScaler()
      scaler.fit(X_train,y_train)
      scaled_x_train=scaler.transform(X_train)
      scaled_x_test=scaler.transform(X_test)

      clf.fit(scaled_x_train,y_train)
      y_pred=clf.predict(scaled_x_test)
      acc1=accuracy_score(y_test,y_pred)
      acc2=f1_score(y_test,y_pred)
      acc3=roc_auc_score(y_test,y_pred)
      cross_val=np.mean(cross_val_score(clf,scaled_x_train,y_train,cv=10))
      return acc1, acc2, acc3, cross_val

    def clf_log(df,clf):

      X=df.loc[:,df.columns[1:]].values
      y=df.loc[:,"diagnosis"].values
      X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
      scaler=StandardScaler()
      scaler.fit(X_train,y_train)
      scaled_x_train=scaler.transform(X_train)
      scaled_x_test=scaler.transform(X_test)

      clf.fit(scaled_x_train,y_train)
      y_pred=clf.predict(scaled_x_test)
      y_prob=clf.predict_proba(scaled_x_test)
      acc1=accuracy_score(y_test,y_pred)
      acc2=f1_score(y_test,y_pred)
      acc3=roc_auc_score(y_test,y_pred)
      cross_val=np.mean(cross_val_score(clf,scaled_x_train,y_train,cv=10))
      return acc1, acc2, acc3,cross_val


    all_eval_metrics={"Model":[],"train_accuracy":[], "Accuracy_score":[], "F1_score":[], "Roc_auc_score":[],"Cross_val_score":[]}

    #decision tree
    print("Decision Tree")
    dict1={"depth":[],"accuracy_score":[],"f1_score":[], "roc_auc_score":[],"cross_val_score":[]}
    for depth in [1,2,3,4,5,6,7,8,9,10,20,50]:
      dt=DecisionTreeClassifier(max_depth=depth)
      acc1, acc2, acc3, cross_val=clf(df,dt)
      dict1["depth"].append(depth)
      dict1["accuracy_score"].append(acc1)
      dict1["f1_score"].append(acc2)
      dict1["roc_auc_score"].append(acc3)
      dict1["cross_val_score"].append(cross_val)
    eval_df=pd.DataFrame(dict1)
    # print(eval_df)
    df2=eval_df[eval_df['cross_val_score']==eval_df.cross_val_score.max()]
    #print("best_model")
    df2_depth=df2["depth"]
    #print(df2)
    for i in df2_depth:
      x=i
    print(x)
    dt=DecisionTreeClassifier(max_depth=x)
    acc1, acc2, acc3, cross_val=clf(df,dt)
    all_eval_metrics["Model"].append("Decision_tree")
    all_eval_metrics["Accuracy_score"].append(acc1)
    all_eval_metrics["F1_score"].append(acc2)
    all_eval_metrics["Roc_auc_score"].append(acc3)
    all_eval_metrics["Cross_val_score"].append(cross_val)
    train_acc=dt.score(scaled_x_train,y_train)
    all_eval_metrics["train_accuracy"].append(train_acc)
    print("Accuracy_score: ",acc1,"F1_score: ",acc2,"AUROC: ",acc3, "Cross_val: ",cross_val)

    #knn
    print("Knn model")
    dict1={"K":[],"accuracy_score":[],"f1_score":[],"roc_auc_score":[],"cross_val_score":[]}
    for k in [1,2,3,4,5,6,7,8,9,10,20,50]:
      knn=KNeighborsClassifier(k)
      acc1, acc2, acc3, cross_val=clf(df,knn)
      dict1["K"].append(k)
      dict1["accuracy_score"].append(acc1)
      dict1["f1_score"].append(acc2)
      dict1["roc_auc_score"].append(acc3)
      dict1["cross_val_score"].append(cross_val)
    eval_df=pd.DataFrame(dict1)
    #print(eval_df)
    df2=eval_df[eval_df['cross_val_score']==eval_df.cross_val_score.max()]
    #print("best_model")
    #print(df2)
    df2_k=df2["K"]
    #print(df2)
    for i in df2_k:
      y=i
    print(y)
    knn=KNeighborsClassifier(y)
    acc1, acc2, acc3, cross_val=clf(df,knn)
    all_eval_metrics["Model"].append("KNN")
    all_eval_metrics["Accuracy_score"].append(acc1)
    all_eval_metrics["F1_score"].append(acc2)
    all_eval_metrics["Roc_auc_score"].append(acc3)
    all_eval_metrics["Cross_val_score"].append(cross_val)
    train_acc=knn.score(scaled_x_train,y_train)
    all_eval_metrics["train_accuracy"].append(train_acc)
    print("Accuracy_score: ",acc1,"F1_score: ",acc2,"AUROC: ",acc3, "Cross_val: ",cross_val)


    #LogisticRegression
    print("LogisticRegression model")
    log_reg=LogisticRegression()
    acc1, acc2, acc3, cross_val=clf_log(df,log_reg)
    all_eval_metrics["Model"].append("LogisticRegression")
    all_eval_metrics["Accuracy_score"].append(acc1)
    all_eval_metrics["F1_score"].append(acc2)
    all_eval_metrics["Roc_auc_score"].append(acc3)
    all_eval_metrics["Cross_val_score"].append(cross_val)
    train_acc=log_reg.score(scaled_x_train,y_train)
    all_eval_metrics["train_accuracy"].append(train_acc)
    print("Accuracy_score: ",acc1,"F1_score: ",acc2,"AUROC: ",acc3, "Cross_val: ",cross_val)

    #Random forest
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators = 100, max_depth=3, criterion = 'entropy', random_state = 0)
    rf.fit(scaled_x_train, y_train)
    y_pred=rf.predict(scaled_x_test)
    acc1=accuracy_score(y_test,y_pred)
    acc2=f1_score(y_test,y_pred)
    acc3=roc_auc_score(y_test,y_pred)
    cross_val=np.mean(cross_val_score(rf,scaled_x_train,y_train,cv=10))
    print("RandomForest model")
    all_eval_metrics["Model"].append("Random forest")
    all_eval_metrics["Accuracy_score"].append(acc1)
    all_eval_metrics["F1_score"].append(acc2)
    all_eval_metrics["Roc_auc_score"].append(acc3)
    all_eval_metrics["Cross_val_score"].append(cross_val)
    train_acc=rf.score(scaled_x_train,y_train)
    all_eval_metrics["train_accuracy"].append(train_acc)
    print("Accuracy_score: ",acc1,"F1_score: ",acc2,"AUROC: ",acc3, "Cross_val: ",cross_val)

    #SVC
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import SVC
    svc = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    svc.fit(scaled_x_train, y_train)
    y_pred=svc.predict(scaled_x_test)
    acc1=accuracy_score(y_test,y_pred)
    acc2=f1_score(y_test,y_pred)
    acc3=roc_auc_score(y_test,y_pred)
    cross_val=np.mean(cross_val_score(svc,scaled_x_train,y_train,cv=10))
    print("SVC model")
    all_eval_metrics["Model"].append("SVC")
    all_eval_metrics["Accuracy_score"].append(acc1)
    all_eval_metrics["F1_score"].append(acc2)
    all_eval_metrics["Roc_auc_score"].append(acc3)
    all_eval_metrics["Cross_val_score"].append(cross_val)
    train_acc=svc.score(scaled_x_train,y_train)
    all_eval_metrics["train_accuracy"].append(train_acc)
    print("Accuracy_score: ",acc1,"F1_score: ",acc2,"AUROC: ",acc3, "Cross_val: ",cross_val)

    #XGBoost (eXtreme Gradient Boosting)
    import xgboost as xgb
    xgb_model = xgb.XGBClassifier().fit(scaled_x_train, y_train)
    y_pred=xgb_model.predict(scaled_x_test)
    acc1=accuracy_score(y_test,y_pred)
    acc2=f1_score(y_test,y_pred)
    acc3=roc_auc_score(y_test,y_pred)
    cross_val=np.mean(cross_val_score(xgb_model,scaled_x_train,y_train,cv=10))
    print("XGBoost model")
    all_eval_metrics["Model"].append("XGBoost")
    all_eval_metrics["Accuracy_score"].append(acc1)
    all_eval_metrics["F1_score"].append(acc2)
    all_eval_metrics["Roc_auc_score"].append(acc3)
    all_eval_metrics["Cross_val_score"].append(cross_val)
    train_acc=xgb_model.score(scaled_x_train,y_train)
    all_eval_metrics["train_accuracy"].append(train_acc)
    print("Accuracy_score: ",acc1,"F1_score: ",acc2,"AUROC: ",acc3, "Cross_val: ",cross_val)

    #GaussianNB
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb.fit(scaled_x_train, y_train)
    y_pred=gnb.predict(scaled_x_test)
    acc1=accuracy_score(y_test,y_pred)
    acc2=f1_score(y_test,y_pred)
    acc3=roc_auc_score(y_test,y_pred)
    cross_val=np.mean(cross_val_score(gnb,scaled_x_train,y_train,cv=10))
    print("GaussianNB model")
    all_eval_metrics["Model"].append("GaussianNB")
    all_eval_metrics["Accuracy_score"].append(acc1)
    all_eval_metrics["F1_score"].append(acc2)
    all_eval_metrics["Roc_auc_score"].append(acc3)
    all_eval_metrics["Cross_val_score"].append(cross_val)
    train_acc=gnb.score(scaled_x_train,y_train)
    all_eval_metrics["train_accuracy"].append(train_acc)
    print("Accuracy_score: ",acc1,"F1_score: ",acc2,"AUROC: ",acc3, "Cross_val: ",cross_val)

    metric_df = pd.DataFrame.from_dict(all_eval_metrics, orient='index')
    metric_df = metric_df.transpose()
    print(metric_df)


![c10](https://github.com/Priyadharshika19/Breast_Cancer_Classification/assets/129640468/6ad6c033-ea84-4789-97ab-0563f0e4bd09)

11.**Feature Importance**

    dt=DecisionTreeClassifier(max_depth=2)
    dt.fit(scaled_x_train,y_train)
    importance=dt.feature_importances_
    dict2={"importance":importance,"features":df.columns}
    importance_feature=pd.DataFrame.from_dict(dict2, orient='index')
    importance_feature = importance_feature.transpose()
    importance_feature[importance_feature["importance"]>0]

![c11](https://github.com/Priyadharshika19/Breast_Cancer_Classification/assets/129640468/7fe3a6d3-fba8-4e6d-b519-1c0575d46740)

**Conclusion**

    With all these models, SVM provides better AUROC accuracy than others.
