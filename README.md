# Loan Status Prediction
> A Support Vector Machine model to predict whether loan can be sanctioned or not

## Prerequisites
- [dataset](https://www.kaggle.com/datasets/ninzaami/loan-predication)
- Numpy 
- Pandas
- Seaborn
- Sklearn

## Process
1. Loading the modules and reading the csv file
2. Checking and clearing the null values in the dataframe ```loan_data.dropna()```
3. Label encoding using dictionary ```loan_data.replace({'Loan_Status':{'N':0,'Y':1}},inplace=True)```
    - 1 for yes
    - 0 for no

4. Checking column values of 'Dependents'
5. Here we got 3+ values as count of 41 so we can't have that many, so
6. Replacing value of 3+ to 4 by ```loan_data=loan_data.replace(to_replace='3+' , value=4)```
7. **Visualising the data** using seaborn
    - Education vs Loan_status
    - Marriage status vs Loan_status
    - Self employed vs Property area

8. Convert categorical text into numerical like 
```loan_dataset = loan_data.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},
                                'Self_Employed':{'No':0,'Yes':1},'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},
                                'Education':{'Graduate':1,'Not Graduate':0}})
```
9. Separating the data and the label
10. **Splitting** data into train and test`
11. Train SVM classifier model by ```classifier = svm.SVC(kernel='linear')```
12. Fit the data by ```classifier.fit(x_train,y_train)```
13. Model evaluation by comparing accuracy score of training and test data
14. Prediction part
```
input_data = (1,0,2,0,0,3000,250.0,200.0,180.0,0.0,2)

input_data_as_numpy_array = np.asarray(input_data)

#model predicting only one instance
input_data_reshaped= input_data_as_numpy_array.reshape(1,-1)  
prediction = classifier.predict(input_data_reshaped)

print(prediction)
if(prediction[0] == 0):
  print('Not Sanctioned')
else:
  print('Loan Sanctioned')  
```
