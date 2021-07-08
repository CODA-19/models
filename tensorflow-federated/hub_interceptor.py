from grpc import UnaryUnaryClientInterceptor, UnaryStreamClientInterceptor
import jsonpickle
from google.protobuf.json_format import MessageToJson
from google.protobuf.json_format import Parse

from interceptor import Interceptor
import logging
import json
import importlib

from tensorflow_federated.proto.v0 import executor_pb2
from tensorflow_federated.proto.v0 import executor_pb2_grpc

class HubInterceptor(Interceptor, UnaryUnaryClientInterceptor,
                         UnaryStreamClientInterceptor):

    def __init__(self, logger):
        super().__init__('1.0.0')
        self.logger = logger

    def intercept_unary_unary(self, continuation, client_call_details, request):

        request_path = client_call_details.method
        request_method = request_path.split("/")[-1]

        request = self._pass_through(request_method, request)
        response = continuation(client_call_details, request)
        
        if self.logger.isEnabledFor(logging.WARNING):
            self._log_request(client_call_details, request, response)

        return response

    def intercept_unary_stream(self, continuation, client_call_details,
                               request):

        def on_rpc_complete(response_future):
            if self.logger.isEnabledFor(logging.WARNING):
                self._log_request(client_call_details, request, response_future)
        
        request = self._pass_through(request_method, request)
        response = continuation(client_call_details, request)

        response.add_done_callback(on_rpc_complete)

        return response

    def _get_trailing_metadata(self, response):
        try:
            trailing_metadata = response.trailing_metadata()

            if not trailing_metadata:
                return self.get_trailing_metadata_from_interceptor_exception(
                    response.exception())

            return trailing_metadata
        except AttributeError:
            return self.get_trailing_metadata_from_interceptor_exception(
                response.exception())

    def _get_initial_metadata(self, client_call_details):
        return getattr(client_call_details, 'metadata', tuple())

    def _get_call_method(self, client_call_details):
        return getattr(client_call_details, 'method', None)
    
    def _parse_exception_to_str(self, exception):
        try:
            return self.format_json_object(json.loads(
                exception.debug_error_string()))
        except (AttributeError, ValueError):
            return '{}'

    def _get_fault_message(self, exception):
        try:
            return exception.failure.errors[0].message
        except AttributeError:
            try:
                return exception.details()
            except AttributeError:
                return None

    def _log_successful_request(self, method, metadata_json,
                                request_id, request, trailing_metadata_json,
                                response):
        
        result = '%s' % response.result()

        #print('Successful request: ')
        #print('Method: %s' % method)
        #print('Request: %s' % request)
        #print('Response: %s' % result)

        return

    def _log_failed_request(self, method, metadata_json,
                            request_id, request, trailing_metadata_json,
                            response):

        exception = self._get_error_from_response(response)
        exception_str = self._parse_exception_to_str(exception)
        fault_message = self._get_fault_message(exception)

        print('FAILED REQUEST')

        return

    def _log_request(self, client_call_details, request, response):
        method = self._get_call_method(client_call_details)
        initial_metadata = self._get_initial_metadata(client_call_details)
        initial_metadata_json = self.parse_metadata_to_json(initial_metadata)
        trailing_metadata = self._get_trailing_metadata(response)
        request_id = self.get_request_id_from_metadata(trailing_metadata)
        trailing_metadata_json = self.parse_metadata_to_json(trailing_metadata)

        if response.exception():
            self._log_failed_request(
                method, initial_metadata_json, request_id, request,
                trailing_metadata_json, response)
        else:
            self._log_successful_request(
                method, initial_metadata_json, request_id, request,
                trailing_metadata_json, response)

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
