# https://realpython.com/python-microservices-grpc/
from concurrent import futures
import random
import asyncio
import grpc

from tensorflow_federated.proto.v0 import executor_pb2
from tensorflow_federated.proto.v0 import executor_pb2_grpc
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.executors import eager_tf_executor
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_factory
from tensorflow_federated.python.core.impl.executors import executor_serialization
from tensorflow_federated.python.core.impl.executors import executor_service
from tensorflow_federated.python.core.impl.executors import executor_stacks
from tensorflow_federated.python.core.impl.executors import executor_value_base
from tensorflow_federated.python.core.impl.executors import remote_executor
from tensorflow_federated.python.core.impl.types import placements

class ProxyService(executor_service.ExecutorService):

    def set_channel(self, channel):
      self.channel = channel
      
    async def SetCardinalities(
      self,
      request: executor_pb2.SetCardinalitiesRequest,
      context: grpc.ServicerContext,
    ) -> executor_pb2.SetCardinalitiesResponse:
      print('SetCardinalities')
      try:
        cardinality = request.cardinalities[0].cardinality
        self._executor = remote_executor.RemoteExecutor(self.channel)
        self._executor.set_cardinalities({placements.CLIENTS: cardinality})

        return executor_pb2.SetCardinalitiesResponse()
      except (ValueError, TypeError) as err:
        _set_invalid_arg_err(context, err)
        return executor_pb2.SetCardinalitiesResponse()
      
    def ClearExecutor(
        self, *args, **kwargs
    ) -> executor_pb2.ClearExecutorResponse:
      print('ClearExecutor')
      return super().ClearExecutor(*args, **kwargs)
      
    def CreateValue(
        self, *args, **kwargs
    ) -> executor_pb2.CreateValueResponse:
      print('CreateValue')
      return super().CreateValue(*args, **kwargs)
      
    def CreateCall(
        self, *args, **kwargs
    ) -> executor_pb2.CreateCallResponse:
      print('CreateCall')
      return super().CreateCall(*args, **kwargs)
    
    def CreateSelection(
        self, *args, **kwargs
    ) -> executor_pb2.CreateSelectionResponse:
      print('CreateSelection')
      return super().CreateSelection(*args, **kwargs)
    
    def Compute(
        self, *args, **kwargs
    ) -> executor_pb2.ComputeResponse:
      print('Compute')
      return super().Compute(*args, **kwargs)


def serve(ip_address, port):
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    mirror_port = 8080 - (port - 8080)
    channel = grpc.insecure_channel('{}:{}'.format(ip_address, mirror_port))

    ex_factory = executor_stacks.ResourceManagingExecutorFactory(
        lambda _: eager_tf_executor.EagerTFExecutor())
    service = ProxyService(ex_factory=ex_factory)
    service.set_channel(channel)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    executor_pb2_grpc.add_ExecutorServicer_to_server(service, server)
    
    server.add_insecure_port('[::]:{}'.format(port))
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve('127.0.0.1', 8081)
