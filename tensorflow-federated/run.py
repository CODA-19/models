import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

# Helper functions
def preprocess(dataset, batch_size):

  # Flatten a batch of EMNIST data and return a (features, label) tuple.
  def batch_format_fn(element):
    """"""
    return (tf.reshape(element['pixels'], [-1, 784]), 
            tf.reshape(element['label'], [-1, 1]))

  return dataset.batch(batch_size).map(batch_format_fn)

# Helper functions
def evaluate(server_state):

  keras_model = create_keras_model()
  keras_model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]  
  )
  
  keras_model.set_weights(server_state)
  keras_model.evaluate(central_emnist_test)

# Constants
NUM_CLIENTS = 10
BATCH_SIZE = 20

# Get the model structure JSON file.
with open('model.json','r') as file:
  MODEL_JSON_CONFIG = file.read()

# Create the training data.
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

# Create the clients
client_ids = np.random.choice(emnist_train.client_ids, size=NUM_CLIENTS, replace=False)

# Shard the training data.
federated_train_data = [preprocess(emnist_train.create_tf_dataset_for_client(x), BATCH_SIZE)
  for x in client_ids
]

def create_keras_model():
  return tf.keras.models.model_from_json(MODEL_JSON_CONFIG)

  #return tf.keras.models.Sequential([
  #    tf.keras.layers.InputLayer(input_shape=(784,)),
  #    tf.keras.layers.Dense(10, kernel_initializer='zeros'),
  #    tf.keras.layers.Softmax(),
  #])

def model_fn():
  keras_model = create_keras_model()

  return tff.learning.from_keras_model(
      keras_model,
      input_spec=federated_train_data[0].element_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


# Initializes the hub by instanciating the model function.
@tff.tf_computation
def server_init():
  model = model_fn()
  return model.trainable_variables

# TF - Updates the server model weights as the average of the client model weights.
@tf.function
def server_update(model, mean_client_weights):
  model_weights = model.trainable_variables
  # Assign the mean client weights to the server model.
  tf.nest.map_structure(lambda x, y: x.assign(y),
                        model_weights, mean_client_weights)
  return model_weights

model_weights_type = server_init.type_signature.result

# TFF - This is the tff.tf_computation version of the server update.
@tff.tf_computation(model_weights_type)
def server_update_fn(mean_client_weights):
  model = model_fn()
  return server_update(model, mean_client_weights)

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

model = model_fn()
tf_dataset_type = tff.SequenceType(model.input_spec)

@tff.tf_computation(tf_dataset_type, model_weights_type)
def client_update_fn(tf_dataset, server_weights):
  model = model_fn()
  client_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
  return client_update(model, tf_dataset, server_weights, client_optimizer)

@tff.federated_computation
def initialize_fn():
  return tff.federated_value(server_init(), tff.SERVER)

federated_server_type = tff.FederatedType(model_weights_type, tff.SERVER)
federated_dataset_type = tff.FederatedType(tf_dataset_type, tff.CLIENTS)

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


federated_algorithm = tff.templates.IterativeProcess(
    initialize_fn=initialize_fn,
    next_fn=next_fn
)

print(str(federated_algorithm.initialize.type_signature))
print(str(federated_algorithm.next.type_signature))

central_emnist_test = emnist_test.create_tf_dataset_from_all_clients().take(1000)
central_emnist_test = preprocess(central_emnist_test, BATCH_SIZE)

server_state = federated_algorithm.initialize()
evaluate(server_state)

for round in range(100):
  print(round)
  server_state = federated_algorithm.next(server_state, federated_train_data)
  evaluate(server_state)

evaluate(server_state)