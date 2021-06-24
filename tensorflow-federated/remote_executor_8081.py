import tensorflow_federated as tff

executor_factory = tff.framework.local_executor_factory(max_fanout=100)
tff.simulation.run_server(executor_factory, 10, 8081, None)