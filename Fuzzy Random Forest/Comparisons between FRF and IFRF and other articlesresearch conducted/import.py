'''
please 'pip install ucimlrepo' in terminal to install the package before running this code.

Datasets:
    -Dermatology (ID: 33)
    -Iris (ID: 53)
    -Glass Identification (ID: 42)
    -Breast Cancer Wisconsin (Original) (ID: 15)
    -Statlog (Heart) (ID: 145)
    -Wine (ID: 109)
    -Zoo (ID: 111)
    -Statlog (Australian Credit Approval) (ID: 143)
    -Hepatitis (ID: 46)
    -Ionosphere (ID: 52)
'''

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
dermatology = fetch_ucirepo(id=33) 
  
# data (as pandas dataframes) 
X_dermatology = dermatology.data.features 
y_dermatology = dermatology.data.targets 
  
# metadata 
print(dermatology.metadata) 
  
# variable information 
print(dermatology.variables) 

# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
X_iris = iris.data.features 
y_iris = iris.data.targets 
  
# metadata 
print(iris.metadata) 
  
# variable information 
print(iris.variables) 

  
# fetch dataset 
glass_identification = fetch_ucirepo(id=42) 
  
# data (as pandas dataframes) 
X_glass = glass_identification.data.features 
y_glass = glass_identification.data.targets 
  
# metadata 
print(glass_identification.metadata) 
  
# variable information 
print(glass_identification.variables) 


# fetch dataset 
breast_cancer_wisconsin_original = fetch_ucirepo(id=15) 
  
# data (as pandas dataframes) 
X_breast_cancer_wisconsin_original = breast_cancer_wisconsin_original.data.features 
y_breast_cancer_wisconsin_original = breast_cancer_wisconsin_original.data.targets 
  
# metadata 
print(breast_cancer_wisconsin_original.metadata) 
  
# variable information 
print(breast_cancer_wisconsin_original.variables) 

# fetch dataset 
statlog_heart = fetch_ucirepo(id=145) 
  
# data (as pandas dataframes) 
X_statlog_heart = statlog_heart.data.features 
y_statlog_heart = statlog_heart.data.targets 
  
# metadata 
print(statlog_heart.metadata) 
  
# variable information 
print(statlog_heart.variables) 

# fetch dataset 
wine = fetch_ucirepo(id=109) 
  
# data (as pandas dataframes) 
X_wine = wine.data.features 
y_wine = wine.data.targets 
  
# metadata 
print(wine.metadata) 
  
# variable information 
print(wine.variables) 

# fetch dataset 
zoo = fetch_ucirepo(id=111) 
  
# data (as pandas dataframes) 
X_zoo = zoo.data.features 
y_zoo = zoo.data.targets 
  
# metadata 
print(zoo.metadata) 
  
# variable information 
print(zoo.variables) 

# fetch dataset 
statlog_australian_credit_approval = fetch_ucirepo(id=143) 
  
# data (as pandas dataframes) 
X_statlog_australian_credit_approval = statlog_australian_credit_approval.data.features 
y_statlog_australian_credit_approval = statlog_australian_credit_approval.data.targets 
  
# metadata 
print(statlog_australian_credit_approval.metadata) 
  
# variable information 
print(statlog_australian_credit_approval.variables) 

# fetch dataset 
hepatitis = fetch_ucirepo(id=46) 
  
# data (as pandas dataframes) 
X_hepatitis = hepatitis.data.features 
y_hepatitis = hepatitis.data.targets 
  
# metadata 
print(hepatitis.metadata) 
  
# variable information 
print(hepatitis.variables) 

# fetch dataset 
ionosphere = fetch_ucirepo(id=52) 
  
# data (as pandas dataframes) 
X_ionosphere = ionosphere.data.features 
y_ionosphere = ionosphere.data.targets 
  
# metadata 
print(ionosphere.metadata) 
  
# variable information 
print(ionosphere.variables) 