import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from plot_helpers import plot_net_history

# you can ignore this line. all im doing is disabling my GPU from running this small net
# small nets run faster on the CPU becuase of all the memory managment and transfer to the GPU
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def create_model() -> Model:
    '''
    This function will use Keras which comes with tensorflow to create a ML model
    Keras is a high level api that allows you to quickly create great ML models

    returns a Keras model
    '''
    input_layer = Input(shape=(20))
    hidden_layer = Dense(10, activation='relu')(input_layer)
    hidden_layer = Dense(5, activation='relu')(hidden_layer)
    output = Dense(1, activation='sigmoid')(hidden_layer)
    model = Model(input_layer,output)
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,loss='binary_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    # read in the data into a pandas Dataframe
    df = pd.read_csv('water_quality.csv') # df is a common variable name for a pandas Dataframe
    # divide the data into features and labels
    # features are just another name for column headers or inputs into a neural network
    data = df.values[:,:-1]
    # normalize the data which in this case means scale all the features from 0 to 1
    # THIS IS IMPORTANT if you do not normalize your data in some way 
    # then the gradient of the larger features will dominate the updated weights
    data = MinMaxScaler().fit_transform(data)
    # grabs all the classes 
    labels = df.values[:,-1]

    
    # split the data into training and testing sets
    # in order to evaluate if your model is good you need to set aside some data that is never
    # exposed to the model. This way when training is done you can test the model and see if the model
    # generalizes well to the data that the model has not seen. This is how you tell if you are overfitting or underfitting 
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, labels, test_size=0.2, random_state=0)

    # ***********TRAINING*********
    # get the model we made
    model = create_model()
    EPOCHS = 100
    BATCH_SIZE = 32
    # this line trains the model and assigns all the metric history to a variable
    history = model.fit(Xtrain,Ytrain,batch_size=BATCH_SIZE,epochs=EPOCHS,validation_split=0.3)

    # ********EVALUATE MODEL********
    # evaluating a model generally means testing the model on the test set and gathering metrics
    # like accuracy, precision or recall. To keep it simple let's just focus on accuracy
    predicted_results = model.predict(Xtest)
    # we used the sigmoid function as an output so the output will be between 0 and 1
    # the problem we are solving is binary so anything above .5 is a 1 and below is a 0
    predicted_results = np.where(predicted_results<0.5,0,1)
    accuracy_result = accuracy_score(Ytest, predicted_results)
    print(f'Accuracy: {accuracy_result}')
    plot_net_history(history.history)


    