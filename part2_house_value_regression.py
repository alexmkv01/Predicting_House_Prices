import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(13,32),
            nn.ReLU(),
            nn.Linear(32,10),
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
        self.median_values = None
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

            """ 
            print("mean:")
            print(self.mean)
            print("")
            print("std:")
            print(self.std)
            print()
            print("median:")
            print(self.median_values)
            print()
            print("mode:")
            print(self.mode)
            print()
            print("columns:")
            print(self.columns)
            print()
            """

        else:
            
            # Separate Numerical and Categorical Columns.
            x_without_ocean_proximity = x.drop('ocean_proximity', axis=1)

            # Fill missing numerical values with median + normalize
            x_without_ocean_proximity = x_without_ocean_proximity.fillna(self.median_values)
            x_without_ocean_proximity = (x_without_ocean_proximity - self.mean) / self.std

            # Fill missing categorical values with mode 
            x['ocean_proximity'].fillna(self.mode)
            one_hot_encoded_ocean_proximities = pd.get_dummies(x['ocean_proximity'])

            # Concatenate one-hot encoded columns and numerical
            x = pd.concat([x_without_ocean_proximity, one_hot_encoded_ocean_proximities], axis=1)

            differences = list(set(self.columns) - set(x.columns.values))

            for col in differences:
                x[col]= 0

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

        # Instantiate model.
        model = Net()

        # Define hyperparameters for learning.
        epochs = 100
        learning_rate = 0.001
        batch_size = 64
        criterion = nn.MSELoss()

        # Separate training data (90% of full dataset) to give us a total 80% train, 10% test, 10% val split.
        x_train, x_val, y_train, y_val = train_test_split(x, y, random_state=104, test_size=0.11, shuffle=True)

        # Preprocess training and validation data.
        x_train, y_train = self._preprocessor(x_train, y_train, training = True) 
        x_val, y_val = self._preprocessor(x_val, y_val, training = False)

        # Convert training data and associated labels to tensors.
        x_train = torch.tensor(x_train.values, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32)

        # Convert validation data and associated labels to tensors.
        x_val = torch.tensor(x_val.values, dtype=torch.float32)
        y_val = torch.tensor(y_val.values, dtype=torch.float32)

        # Combine the training set and training label tensors into a single tensor object.
        train_set = TensorDataset(x_train, y_train)

        # Combine the validation set and validation label tensors into a single tensor object.
        val_set = TensorDataset(x_val, y_val)

        # Batch (already shuffled) data.
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=2) 
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

        # Use stochastic gradient descent optimiser.
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Loop for given number of epochs
        for epoch in range(epochs):  

            training_losses = []
            val_losses = []
            running_loss = 0

            # Execute learning cycle for all batches.
            model.train()
            for i, data in enumerate(train_loader):
                
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # Compute the model output over all inputs in this batch.
                predictions = model(inputs)

                # Compute the MSE loss between the model outputs and labels
                RMSE_loss = torch.sqrt(criterion(predictions, labels))

                running_loss += RMSE_loss.item()

                # Compute the gradients.
                RMSE_loss.backward()

                # Backpropagate the gradients.
                optimizer.step()

            training_loss = running_loss / len(train_loader)
            training_losses.append(training_loss)
            
            running_loss = 0

            model.eval()
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    inputs, labels = data
                    predictions = model(inputs)
                    RMSE_loss = torch.sqrt(criterion(predictions, labels))
                    running_loss += RMSE_loss.item()
            
            val_loss = running_loss / len(val_loader)
            val_losses.append(val_loss)

            print("epoch: {}, training loss: {}".format(epoch+1, training_loss))
            print("epoch: {}, validation loss: {}".format(epoch+1, val_loss))
            print()

        return 

            
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

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=104, train_size=0.9, shuffle=True)

    regressor.fit(x_train, y_train)
    #save_regressor(regressor)
    # Error
    #error = regressor.score(x_train, y_train)
    #print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()

