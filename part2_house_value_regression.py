import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(9,32),
            nn.ReLU(),
            nn.Linear(32,128),
            nn.ReLU(),
            nn.Linear(128,10),
            nn.ReLU(),
            nn.Linear(10,1)
        )

    def forward(self, x):
        output = self.linear_stack(x)
        return output


class Regressor():

    def __init__(self, x, nb_epoch = 1000):
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests

        # Constants used for normalising the numerical features.
        self.mean = None
        self.median = None
        self.std = None
        self.columns = None
        self.mode = None

        # Call preprocessor to get shape of cleaned data.
        X, _ = self._preprocessor(x, training = True)

        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch 
        return


    def _preprocessor(self, x, y = None, training = True):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """
        # x consists of training data. Preprocess.
        if training:

            # Separate numerical and categorical Columns.
            x_without_ocean_proximity = x.drop('ocean_proximity', axis=1)

            # Fill missing numerical values with median + normalize
            self.mean = x_without_ocean_proximity.mean()
            self.std = x_without_ocean_proximity.std()
            self.median_values = x_without_ocean_proximity.median()

            x_without_ocean_proximity = x_without_ocean_proximity.fillna(self.median_values)
            x_without_ocean_proximity = (x_without_ocean_proximity - self.mean) / self.std

            # Fill missing categorical values with mode 
            self.mode = x['ocean_proximity'].mode()
            x['ocean_proximity'].fillna(self.mode)
            one_hot_encoded_ocean_proximities = pd.get_dummies(x['ocean_proximity'])

            # Concatenate one-hot encoded columns and numerical
            x = pd.concat([x_without_ocean_proximity, one_hot_encoded_ocean_proximities], axis=1)

            # Save the column headers (ensuring the test dataset has the same columns)
            self.columns = list(x.columns.values)
            x = x[self.columns]

        else:
            
            # Separate Numerical and Categorical Columns.
            x_without_ocean_proximity = x.drop('ocean_proximity', axis=1)

            # Fill missing numerical values with median + normalize
            x_without_ocean_proximity = x_without_ocean_proximity.fillna(self.median_values)
            x_without_ocean_proximity = (x_without_ocean_proximity - self.mean) / self.std

            # Fill missing categorical values with mode 
            x['ocean_proximity'].fillna(self.mode)
            one_hot_encoded_ocean_proximities = pd.get_dummies(x['ocean_proximity'])

            # Fill missing categorical values with mode 
            self.mode = x['ocean_proximity'].mode()
            x['ocean_proximity'].fillna(self.mode)
            one_hot_encoded_ocean_proximities = pd.get_dummies(x['ocean_proximity'])

            # Concatenate one-hot encoded columns and numerical
            x = pd.concat([x_without_ocean_proximity, one_hot_encoded_ocean_proximities], axis=1)

            # Save the column headers (ensuring the test dataset has the same columns)
            x = x[self.columns]

        # Return preprocessed x and y, return None for y if it was None
        return x, (y if isinstance(y, pd.DataFrame) else None)

        
        
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """
        learning_rate = 0.1
        batch_size = 32


        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget
        trainloader = torch.utils.data.DataLoader(X, batch_size=batch_size,
                                          shuffle=False, num_workers=2)
        
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


        net = Net()
        for epoch in range(2):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
            
            with torch.no_

        return self

            
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget
        pass

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        return 0 # Replace this code with your own

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def RegressorHyperParameterSearch(): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    return  # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x, nb_epoch = 10)


    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=104, train_size=0.8, shuffle=True)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, random_state=104, train_size=0.5, shuffle=True)

    regressor.fit(x_train, y_train)
    #save_regressor(regressor)
    # Error
    #error = regressor.score(x_train, y_train)
    #print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()

