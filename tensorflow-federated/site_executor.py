import sys
import tensorflow_federated as tff
import logging

logging.basicConfig(level=logging.DEBUG)

if len(sys.argv) != 2:
   print('Incorrect number of arguments. Invoke as: site_executor.py <port number>')
   exit()

try:
  port = int(sys.argv[1])
except:
   print('Incorrect port number supplied; could not be interpreted as integer.')
   exit()

if port < 7000 or port > 9000:
   print('Incorrect port range. Port should be between 7000 and 9000.')
   exit()

print('Setting up site executor.')

executor_factory = tff.framework.local_executor_factory(max_fanout=100)
tff.simulation.run_server(executor_factory, 10, port, None)

print('Site executor has finished running.')