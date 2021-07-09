from concurrent import futures
import grpc
import json

from tensorflow_federated.proto.v0 import executor_pb2
from tensorflow_federated.proto.v0 import executor_pb2_grpc
from tensorflow_federated.python.core.impl.executors import executor_service
from tensorflow_federated.python.core.impl.executors.executor_stacks import remote_executor_factory

from google.protobuf.json_format import MessageToJson
from google.protobuf.json_format import Parse

class ProxyService(executor_service.ExecutorService):

    def _pass_through(self, request_method, request):
        
      request_class = ''

      if request_method == 'ClearExecutor':
        request_class = 'executor_pb2.ClearExecutorRequest'
      elif request_method == 'SetCardinalities':
        request_class = 'executor_pb2.SetCardinalitiesRequest'
      elif request_method == 'CreateValue':
        request_class = 'executor_pb2.CreateValueRequest'
      elif request_method == 'CreateStruct':
        request_class = 'executor_pb2.CreateStructRequest'
      elif request_method == 'CreateCall':
        request_class = 'executor_pb2.CreateCallRequest'
      elif request_method == 'Compute':
        request_class = 'executor_pb2.ComputeRequest'
      elif request_method == 'Dispose':
        request_class = 'executor_pb2.DisposeRequest'
      else:
        print('Unrecognized request method: %s' % (request_path))
        exit()
      
      message_serialized = MessageToJson(request)

      request_serialized = {
          'request_class': request_class,
          'message_serialized': message_serialized
      }
    
      request_serialized_json = json.dumps(request_serialized)
      request_parsed = json.loads(request_serialized_json)

      request_class_obj = eval(request_parsed['request_class'])
      request_message_obj = request_parsed['message_serialized']

      request_deserialized = Parse(request_message_obj, request_class_obj())

      return request_deserialized
      
    def ClearExecutor(
      self,
      request: executor_pb2.ClearExecutorRequest,
      context: grpc.ServicerContext,
    ) -> executor_pb2.ClearExecutorResponse:
      print('ClearExecutor')
      return super().ClearExecutor(self._pass_through('ClearExecutor', request), context)
      
    def CreateValue(
      self,
      request: executor_pb2.CreateValueRequest,
      context: grpc.ServicerContext,
    ) -> executor_pb2.CreateValueResponse:
      print('CreateValue')
      return super().CreateValue(self._pass_through('CreateValue', request), context)
      
    def CreateCall(
      self,
      request: executor_pb2.CreateCallRequest,
      context: grpc.ServicerContext,
    ) -> executor_pb2.CreateCallResponse:
      print('CreateCall')
      return super().CreateCall(self._pass_through('CreateCall', request), context)
    
    def CreateStruct(
      self,
      request: executor_pb2.CreateStructRequest,
      context: grpc.ServicerContext,
    ) -> executor_pb2.CreateStructResponse:
      print('CreateStruct')
      return super().CreateStruct(self._pass_through('CreateStruct', request), context)

    def CreateSelection(
      self,
      request: executor_pb2.CreateSelectionRequest,
      context: grpc.ServicerContext,
    ) -> executor_pb2.CreateSelectionResponse:
      print('CreateSelection')
      return super().CreateSelection(self._pass_through('CreateSelection', request), context)
    
    def Compute(
      self,
      request: executor_pb2.ComputeRequest,
      context: grpc.ServicerContext,
    ) -> executor_pb2.ComputeResponse:
      print('Compute')
      return super().Compute(self._pass_through('Compute', request), context)


def serve(ip_address, port, proxied_port):
    
    channel = grpc.insecure_channel('{}:{}'.format(ip_address, proxied_port))

    ex_factory = remote_executor_factory(channels=[channel])
    service = ProxyService(ex_factory=ex_factory)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    executor_pb2_grpc.add_ExecutorServicer_to_server(service, server)
    
    server.add_insecure_port('[::]:{}'.format(port))
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve('127.0.0.1', 8083, 8084)
