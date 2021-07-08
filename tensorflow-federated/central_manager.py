import collections
import grpc
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from grpc import _channel

from logging_interceptor import LoggingInterceptor

import json
import sys
import logging

# 
#  Configure logger to print to STDOUT
# 

logger = logging.getLogger()
#root.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# 
#  Configure training task, preprocess data, plan task (/prepare)
# 

NUM_CLIENTS = 2
BATCH_SIZE = 20

OPTIMIZER = 'SGD'
LOSS_FUNCTION = 'SparseCategoricalCrossentropy'
METRIC_FUNCTION = 'SparseCategoricalAccuracy'

# Get the model structure JSON
with open('model.json','r') as file:
  MODEL_JSON_CONFIG = file.read()

# Preprocess EMNIST data.
def preprocess(dataset, batch_size):

  # Flatten a batch of EMNIST data and return a (features, label) tuple.
  def batch_format_fn(element):
    return (tf.reshape(element['pixels'], [-1, 784]), 
            tf.reshape(element['label'], [-1, 1]))

  return dataset.batch(batch_size).map(batch_format_fn)

# Create the training data.
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

# Create the clients
client_ids = np.random.choice(emnist_train.client_ids, size=NUM_CLIENTS, replace=False)

# Shard the training data.
federated_train_data = [preprocess(emnist_train.create_tf_dataset_for_client(x), BATCH_SIZE)
  for x in client_ids
]

# Setup the remote executors
ip_address = '127.0.0.1'
port = 8080

channels = []
for i in range(NUM_CLIENTS):
  print('Setting up channel %i' % i)
  real_channel = grpc.insecure_channel('{}:{}'.format(ip_address, port+i+1))
  intercept_channel = grpc.intercept_channel(real_channel, LoggingInterceptor(logger))
  channels.append(intercept_channel)

factory = tff.framework.remote_executor_factory(channels)
context = tff.framework.ExecutionContext(factory)
tff.framework.set_default_context(context)

#tff.backends.native.set_remote_execution_context(channels)

# 
#  Keras helper functions
# 

# Create the Keras model.
def create_keras_model(model_json_config):
  return tf.keras.models.model_from_json(model_json_config)

# Run the Keras model.
def model_fn(model_json_config, loss_function, metric_function):
  keras_model = create_keras_model(model_json_config)

  return tff.learning.from_keras_model(
      keras_model,
      input_spec=federated_train_data[0].element_spec,
      loss=getattr(tf.keras.losses, loss_function)(),
      metrics=[getattr(tf.keras.metrics, metric_function)()])

# Evaluate the Keras model.
def evaluate(model_json_config, server_state):

  keras_model = create_keras_model(model_json_config)
  keras_model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]  
  )
  
  keras_model.set_weights(server_state)
  keras_model.evaluate(central_emnist_test)

# 
#  TFF initialization
# 

# Initializes the server by instanciating the model function.
@tff.tf_computation
def server_init():
  model = model_fn(MODEL_JSON_CONFIG, LOSS_FUNCTION, METRIC_FUNCTION)
  return model.trainable_variables

# TFF - Create model and do some introspection for type definition.
model = model_fn(MODEL_JSON_CONFIG, LOSS_FUNCTION, METRIC_FUNCTION)
model_weights_type = server_init.type_signature.result
tf_dataset_type = tff.SequenceType(model.input_spec)
federated_dataset_type = tff.FederatedType(tf_dataset_type, tff.CLIENTS)
federated_server_type = tff.FederatedType(model_weights_type, tff.SERVER)

# 
#  TF + TFF server updates
# 

# TF - Updates the server model weights as the average of the client model weights.
@tf.function
def server_update(model, mean_client_weights):
  model_weights = model.trainable_variables
  # Assign the mean client weights to the server model.
  tf.nest.map_structure(lambda x, y: x.assign(y),
                        model_weights, mean_client_weights)
  return model_weights

# TFF - This is the tff.tf_computation version of the server update.
@tff.tf_computation(model_weights_type)
def server_update_fn(mean_client_weights):
  model = model_fn(MODEL_JSON_CONFIG, LOSS_FUNCTION, METRIC_FUNCTION)
  return server_update(model, mean_client_weights)

# 
#  TF and TFF client updates.
# 

# TF - Performs training (using the server model weights) on the client's dataset.
@tf.function
def client_update(model, dataset, server_weights, client_optimizer):
  # Initialize the client model with the current server weights.
  client_weights = model.trainable_variables
  # Assign the server weights to the client model.
  tf.nest.map_structure(lambda x, y: x.assign(y),
                        client_weights, server_weights)

  # Use the client_optimizer to update the local model.
  for batch in dataset:
    with tf.GradientTape() as tape:
      # Compute a forward pass on the batch of data
      outputs = model.forward_pass(batch)

    # Compute the corresponding gradient
    grads = tape.gradient(outputs.loss, client_weights)
    grads_and_vars = zip(grads, client_weights)

    # Apply the gradient using a client optimizer.
    client_optimizer.apply_gradients(grads_and_vars)

  return client_weights

@tff.tf_computation(tf_dataset_type, model_weights_type)
def client_update_fn(tf_dataset, server_weights):
  model = model_fn(MODEL_JSON_CONFIG, LOSS_FUNCTION, METRIC_FUNCTION)
  client_optimizer = getattr(tf.keras.optimizers, OPTIMIZER)(learning_rate=0.01)
  return client_update(model, tf_dataset, server_weights, client_optimizer)

# 
#  TFF federated computation definition.
# 

@tff.federated_computation
def initialize_fn():
  return tff.federated_value(server_init(), tff.SERVER)

@tff.federated_computation(federated_server_type, federated_dataset_type)
def next_fn(server_weights, federated_dataset):
  # Broadcast the server weights to the clients.
  server_weights_at_client = tff.federated_broadcast(server_weights)

  # Each client computes their updated weights.
  client_weights = tff.federated_map(
      client_update_fn, (federated_dataset, server_weights_at_client))

  # The server averages these updates.
  mean_client_weights = tff.federated_mean(client_weights)

  # The server updates its model.
  server_weights = tff.federated_map(server_update_fn, mean_client_weights)

  return server_weights

# 
#  Get and preprocess testing data.
# 
central_emnist_test = emnist_test.create_tf_dataset_from_all_clients().take(1000)
central_emnist_test = preprocess(central_emnist_test, BATCH_SIZE)

# 
#  Run the federated computation.
# 
federated_algorithm = tff.templates.IterativeProcess(
    initialize_fn=initialize_fn,
    next_fn=next_fn
)

print('Initializing algorithm... ')
server_state = federated_algorithm.initialize()
evaluate(MODEL_JSON_CONFIG, server_state)

# Federated averaging for 100 rounds
for round in range(100):
  print(round)
  server_state = federated_algorithm.next(server_state, federated_train_data)
  evaluate(MODEL_JSON_CONFIG, server_state)

# Do one final evaluation.
evaluate(MODEL_JSON_CONFIG, server_state)