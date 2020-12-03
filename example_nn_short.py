from __future__ import print_function

import math

#from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

folder = "/home/openerp/Scrivania/ricerca_lavoro/ceravolo/giustizia/MinisteroGiustizia/generated/"

my_dataframe = pd.read_csv(folder+"embedded_events.csv", sep=",")

my_dataframe = my_dataframe.reindex(
    np.random.permutation(my_dataframe.index))

def preprocess_features(my_dataframe):
  """Prepares input features from input data set.

  Args:
    my_dataframe: A Pandas DataFrame expected to contain data
      for events and label
  Returns:
    A DataFrame that contains the features to be used for the model, including
    synthetic features if needed.
  """
  selected_features = my_dataframe[
    ["e0","e1","e2","e3","e4","e5","e6","e7","e8","e9","e10","e11"]]
  processed_features = selected_features.copy()
  # Create a synthetic feature if necessary.
  #processed_features["new_feature"] = (
    #my_dataframe["velocity"] ... 
  return processed_features

def preprocess_targets(my_dataframe):
  """Prepares target features (i.e., labels) from input data set.

  Args:
    my_dataframe: A Pandas DataFrame expected to contain data
      for velocity and force.
  Returns:
    A DataFrame that contains the target feature.
  """
  output_targets = pd.DataFrame()
  # Scale the target if necessary
  output_targets["label"] = (
    my_dataframe["label"])  # / 1000.0)
  return output_targets

# Choose the first 90000 (out of 18000) examples for training.
training_examples = preprocess_features(my_dataframe.head(100000))
training_targets = preprocess_targets(my_dataframe.head(100000))

# Choose the last 90000 (out of 180000) examples for validation.
validation_examples = preprocess_features(my_dataframe.tail(100000))
validation_targets = preprocess_targets(my_dataframe.tail(100000))

# Double-check that we've done the right thing.
print("Training examples summary:")
print(training_examples.describe())
print("Validation examples summary:")
print(validation_examples.describe())
print(training_examples)
print("Training targets summary:")
print(training_targets.describe())
print("Validation targets summary:")
print(validation_targets.describe())


#plt.scatter(training_examples.values, training_targets.values)
#plt.scatter(validation_examples.values, validation_targets.values)
#plt.show()


def construct_feature_columns(input_features):
  """Construct the TensorFlow Feature Columns.

  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """ 
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a neural net regression model.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
    
    # Convert pandas data into a dict of np arrays.
    # Normalization needed for cols e0-e3, e8-e11
    features = {key:np.array(value) for key,value in dict(features).items()}                                             
 
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
#    ds1 = ds.filter(lambda self, x: x[42] == 0)
#    ds1 = ds1.batch(int(batch_size/2)).repeat(num_epochs)
#    ds2 = ds.filter(lambda self, x: x[42] != 0)
#    ds2 = ds2.batch(int(batch_size/2)).repeat(num_epochs)
#    ds = ds1.concatenate(ds2)
    ds = ds.batch(batch_size).repeat(num_epochs)
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def train_nn_regression_model(
    learning_rate,
    steps,
    batch_size,
    hidden_units,
    activation_fn,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
  """Trains a neural network regression model.
  
  In addition to training, this function also prints training progress information,
  as well as a plot of the training and validation loss over time.
  
  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    hidden_units: A `list` of int values, specifying the number of neurons in each layer.
    training_examples: A `DataFrame` containing one or more columns from
      `my_dataframe` to use as input features for training.
    training_targets: A `DataFrame` containing exactly one column from
      `my_dataframe` to use as target for training.
    validation_examples: A `DataFrame` containing one or more columns from
      `my_dataframe` to use as input features for validation.
    validation_targets: A `DataFrame` containing exactly one column from
      `my_dataframe` to use as target for validation.
      
  Returns:
    A `DNNRegressor` object trained on the training data.
  """

  periods = 10 # 10
  steps_per_period = steps / periods
  
  # Create a DNNRegressor object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  dnn_regressor = tf.estimator.DNNRegressor(
      feature_columns=construct_feature_columns(training_examples),
      hidden_units=hidden_units, activation_fn=activation_fn,
      optimizer=my_optimizer,
  )
  # this model does not accept a specialized activation function for each level: all levels use the same a.f.
  # Create input functions.
  training_input_fn = lambda: my_input_fn(training_examples, 
                                          training_targets["label"], 
                                          batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(training_examples, 
                                                  training_targets["label"], 
                                                  num_epochs=1, 
                                                  shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                    validation_targets["label"], 
                                                    num_epochs=1, 
                                                    shuffle=False)

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("RMSE (on training data):")
  training_rmse = []
  validation_rmse = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    dnn_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Take a break and compute predictions.
    training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
    training_predictions = np.array([item['predictions'][0] for item in training_predictions])
    
    validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
    validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
    
    # Compute training and validation loss.
    training_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(training_predictions, training_targets))
    validation_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(validation_predictions, validation_targets))
    # Occasionally print the current loss.
    print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
    # Add the loss metrics from this period to our list.
    training_rmse.append(training_root_mean_squared_error)
    validation_rmse.append(validation_root_mean_squared_error)
  print("Model training finished.")

  # Output a graph of loss metrics over periods.
  plt.ylabel("RMSE")
  plt.xlabel("Periods")
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout()
  plt.plot(training_rmse, label="training")
  plt.plot(validation_rmse, label="validation")
  plt.legend()

  print("Final RMSE (on training data):   %0.2f" % training_root_mean_squared_error)
  print("Final RMSE (on validation data): %0.2f" % validation_root_mean_squared_error)

  return dnn_regressor


########## training parameters must be tuned using real data!!! ###########

dnn_regressor = train_nn_regression_model(
    learning_rate=0.02,
    steps=500,     #500,
    batch_size=8,
    hidden_units=[12, 12,12,12,12],
    activation_fn="relu",  #"sigmoid",  #"tanh",   #"relu",
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

my_test_data = pd.read_csv(folder+"embedded_events_test.csv", sep=",")
test_examples = preprocess_features(my_test_data)
test_targets = preprocess_targets(my_test_data)
test_input_fn = lambda: my_input_fn(test_examples, 
                                    test_targets["label"], 
                                    num_epochs=1, 
                                    shuffle=False)
test_predictions1 = dnn_regressor.predict(input_fn=test_input_fn)
test_predictions = np.array([item['predictions'][0] for item in test_predictions1])
root_mean_squared_error = math.sqrt(
    metrics.mean_squared_error(test_predictions, test_targets))

print("Final RMSE (on test data): %0.2f" % root_mean_squared_error)

plt.show()

nw0 = 0
nw1 = 0
for k in range(len(test_predictions)):
    if abs(round(test_predictions[k])-test_targets['label'][k]) > 0.25:
#    if (test_predictions[k] > 0.5 and test_targets['label'][k] == 0) or (test_predictions[k] <= 0.5 and test_targets['label'][k] == 1):
# classification with threshold 0.5
        print(test_predictions[k], test_targets['label'][k])
        if test_targets['label'][k] == 0:
            nw0 = nw0 + 1
        else:
            nw1 = nw1 + 1
print("Wrong 0 labels: " + str(nw0)+", wrong 1 labels: "+str(nw1))
print("Total " + str(len(test_predictions)) + " tests")