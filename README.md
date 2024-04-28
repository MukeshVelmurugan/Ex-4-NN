# EX-04 Implementation of MLP with Backpropagation for Multiclassification</H1>
### Aim: &emsp;
To implement a Multilayer Perceptron for Multi classification &emsp;&emsp;&emsp; &emsp;&emsp;&emsp;&emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp;**DATE:**
### Theory:
A multilayer perceptron (MLP) is a feedforward artificial neural network that generates a set of outputs from a set of inputs. An MLP is characterized by several layers of input nodes connected as a directed graph between the input and output layers. MLP uses back propagation for training the network. MLP is a deep learning method.
A multilayer perceptron is a neural network connecting multiple layers in a directed graph, which means that the signal path through the nodes only goes one way. Each node, apart from the input nodes, has a nonlinear activation function. An MLP uses backpropagation as a supervised learning technique.
MLP is widely used for solving problems that require supervised learning as well as research into computational neuroscience and parallel distributed processing. Applications include speech recognition, image recognition and machine translation.
<table>
<tr>
<td width=40%>

MLP has the following features:<br>
Ø  Adjusts the synaptic weights based on Error Correction Rule<br>
Ø  Adopts LMS<br>
Ø  possess Backpropagation algorithm for recurrent propagation of error<br>
Ø  Consists of two passes<br>
  	(i)Feed Forward pass<br>
	         (ii)Backward pass<br>     
Ø  Learning process –backpropagation<br>
Ø  Computationally efficient method<br>
</td> 
<td>

![image 10](https://user-images.githubusercontent.com/112920679/198804559-5b28cbc4-d8f4-4074-804b-2ebc82d9eb4a.jpg)</td>
</tr> 
</table>

<table>
<tr>
<td width=70%>

3 Distinctive Characteristics of MLP:<br>
Ø  Each neuron in network includes a non-linear activation function.<br>
Ø  Contains one or more hidden layers with hidden neurons.<br>
Ø  Network exhibits high degree of connectivity determined by the synapses of the network.
</td> 
<td>

<img height=40% width=99% src="https://user-images.githubusercontent.com/112920679/198814300-0e5fccdf-d3ea-4fa0-b053-98ca3a7b0800.png">
</td>
</tr> 
</table>





3 Signals involved in MLP are:
 Functional Signal<br>
  ->input signal.  ->propagates forward neuron by neuron thro network and emerges at an output signal.  ->F(x,w) at each neuron as it passes<br>
Error Signal<br>
   ->Originates at an output neuron. ->Propagates backward through the network neuron. ->Involves error dependent function in one way or the other<br>
Each hidden neuron or output neuron of MLP is designed to perform two computations:The computation of the function signal appearing at the output of a neuron which is expressed as a continuous non-linear function of the input signal and synaptic weights associated with that neuron.The computation of an estimate of the gradient vector is needed for the backward pass through the network.<br>
<table>
<tr>
<td width=40%>


TWO PASSES OF COMPUTATION:<br>
In the forward pass:<br>
•       Synaptic weights remain unaltered<br>
•       Function signal are computed neuron by neuron<br>
•       Function signal of jth neuron is
</td> 
<td>
<img height=10% width=39% src="https://user-images.githubusercontent.com/112920679/198814313-2426b3a2-5b8f-489e-af0a-674cc85bd89d.png">
<img height=40% width=79% src="https://user-images.githubusercontent.com/112920679/198814328-1a69a3cd-7e02-4829-b773-8338ac8dcd35.png">
<img height=50% width=99% src="https://user-images.githubusercontent.com/112920679/198814339-9c9e5c30-ac2d-4f50-910c-9732f83cabe4.png">
</td>
</tr> 
</table>



If jth neuron is output neuron, the m=mL  and output of j th neuron is![image](https://user-images.githubusercontent.com/112920679/198814349-a6aee083-d476-41c4-b662-8968b5fc9880.png)
Forward phase begins with in the first hidden layer and end by computing ej(n) in the output layer<img width=20% src="https://user-images.githubusercontent.com/112920679/198814353-276eadb5-116e-4941-b04e-e96befae02ed.png">

<table>
<tr>
<td width=40%>

In the backward pass,<br>
•       It starts from the output layer by passing error signal towards leftward layer neurons to compute local gradient recursively in each neuron<br>
•        it changes the synaptic weight by delta rule
</td> 
<td>

![image](https://user-images.githubusercontent.com/112920679/198814362-05a251fd-fceb-43cd-867b-75e6339d870a.png)
</td>
</tr> 
</table>



<H3>Algorithm:</H3>

1. Import the necessary libraries of python.<br>
2. After that, create a list of attribute names in the dataset and use it in a call to the read_csv() function of the pandas library along with the name of the CSV file containing the dataset.<br>
3. Divide the dataset into two parts. While the first part contains the first four columns that we assign in the variable x. Likewise, the second part contains only the last column that is the class label. Further, assign it to the variable y.<br>
4. Call the train_test_split() function that further divides the dataset into training data and testing data with a testing data size of 20%.
Normalize our dataset. <br>
5. In order to do that we call the StandardScaler() function. Basically, the StandardScaler() function subtracts the mean from a feature and scales it to the unit variance.<br>
6. Invoke the MLPClassifier() function with appropriate parameters indicating the hidden layer sizes, activation function, and the maximum number of iterations.<br>
7. In order to get the predicted values we call the predict() function on the testing data set.<br>
8. Finally, call the functions confusion_matrix(), and the classification_report() in order to evaluate the performance of our classifier.
### Program:
```Python
import pandas as pd								#Developed By: MUKESH V
import sklearn									#Register No : 212222230086
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
irisdata = pd.read_csv(url, names=names)
# Takes first 4 columns and assign them to variable "X"
X = irisdata.iloc[:, 0:4]
# Takes first 5th columns and assign them to variable "Y". Object dtype refers to strings.
y = irisdata.select_dtypes(include=[object])
X.head()
y.head()
# y actually contains all categories or classes:
y.Class.unique()
# Now transforming categorial into numerical values
le = preprocessing.LabelEncoder()
y = y.apply(le.fit_transform)
y.head()
# Now for train and test split (80% of  dataset into  training set and  other 20% into test data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
# Feature scaling
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp.fit(X_train, y_train.values.ravel())
predictions = mlp.predict(X_test)
print(predictions)
# Last thing: evaluation of algorithm performance in classifying flowers
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
```
### Output:
![image](https://github.com/MukeshVelmurugan/Ex-4-NN/assets/118707363/61c8dd5e-54a8-4195-9bb0-7b42383039ff)

### Program:
```Python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
arr = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
df = pd.read_csv(url, names=arr)
print(df.head())
a = df.iloc[:, 0:4]
b = df.select_dtypes(include=[object])
b = df.iloc[:,4:5]
training_a, testing_a, training_b, testing_b = train_test_split(a, b, test_size = 0.25)
myscaler = StandardScaler()
myscaler.fit(training_a)
training_a = myscaler.transform(training_a)
testing_a = myscaler.transform(testing_a)
m1 = MLPClassifier(hidden_layer_sizes=(12, 13, 14), activation='relu', solver='adam', max_iter=2500)
m1.fit(training_a, training_b.values.ravel())
predicted_values = m1.predict(testing_a)
print(confusion_matrix(testing_b,predicted_values))
print(classification_report(testing_b,predicted_values))
```
### Output:
![image](https://github.com/MukeshVelmurugan/Ex-4-NN/assets/118707363/32de68c1-0848-4bf6-a933-114d850384c1)

### Result:
Thus, MLP is implemented for multi-classification using python.
