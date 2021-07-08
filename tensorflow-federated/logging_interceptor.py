from grpc import UnaryUnaryClientInterceptor, UnaryStreamClientInterceptor
import jsonpickle
from google.protobuf.json_format import MessageToJson
from google.protobuf.json_format import MessageToDict
from google.protobuf.json_format import Parse

from interceptor import Interceptor
import logging
import json
import importlib

from tensorflow_federated.proto.v0 import executor_pb2
from tensorflow_federated.proto.v0 import executor_pb2_grpc

class LoggingInterceptor(Interceptor, UnaryUnaryClientInterceptor,
                         UnaryStreamClientInterceptor):

    _FULL_REQUEST_LOG_LINE = ('Request\n-------\nMethod: {}\nHost: {}\n'
                              'Headers: {}\nRequest: {}\n\nResponse\n-------\n'
                              'Headers: {}\nResponse: {}\n')
    _FULL_FAULT_LOG_LINE = ('Request\n-------\nMethod: {}\nHost: {}\n'
                            'Headers: {}\nRequest: {}\n\nResponse\n-------\n'
                            'Headers: {}\nFault: {}\n')
    _SUMMARY_LOG_LINE = ('Request made: Host: {}, '
                         'Method: {}, RequestId: {}, IsFault: {}, '
                         'FaultMessage: {}')

    # Initializer for the LoggingInterceptor.
    # Args: logger: An instance of logging.Logger.
    def __init__(self, logger):
        super().__init__('1.0.0')
        self.logger = logger

    # Retrieves trailing metadata from a response object.
    # Args: response: A grpc.Call/grpc.Future instance.
    # Returns: A tuple of metadatum representing response header key value pairs.
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

    # Retrieves the initial metadata from client_call_details.
    # Args: client_call_details: An instance of grpc.ClientCallDetails.
    # Returns: A tuple of metadatum representing request header key value pairs.
    def _get_initial_metadata(self, client_call_details):
        return getattr(client_call_details, 'metadata', tuple())

    # Retrieves the call method from client_call_details.
    # Args: client_call_details: An instance of grpc.ClientCallDetails.
    # Returns: A str with the call method or None if it isn't present.
    def _get_call_method(self, client_call_details):
        return getattr(client_call_details, 'method', None)
    
    # Parses response exception object to str for logging.
    # Args: exception: A grpc.Call instance.
    # Returns: A str representing a exception from the API.
    def _parse_exception_to_str(self, exception):
        try:
            # if exception.failure isn't present then it's likely this is a
            # transport error with a .debug_error_string method and the
            # returned JSON string will need to be formatted.
            return self.format_json_object(json.loads(
                exception.debug_error_string()))
        except (AttributeError, ValueError):
            # if both attempts to retrieve serializable error data fail
            # then simply return an empty JSON string
            return '{}'

    def _get_fault_message(self, exception):
        """Retrieves a fault/error message from an exception object.

        Returns None if no error message can be found on the exception.

        Returns:
            A str with an error message or None if one cannot be found.

        Args:
            response: A grpc.Call/grpc.Future instance.
            exception: A grpc.Call instance.
        """
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
        """Handles logging all requests.

        Args:
            client_call_details: An instance of grpc.ClientCallDetails.
            request: An instance of a request proto message.
            response: A grpc.Call/grpc.Future instance.
        """
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

    def intercept_unary_unary(self, continuation, client_call_details, request):
        """Intercepts and logs API interactions.

        Overrides abstract method defined in grpc.UnaryUnaryClientInterceptor.

        Args:
            continuation: a function to continue the request process.
            client_call_details: a grpc._interceptor._ClientCallDetails
                instance containing request metadata.
            request: a SearchGoogleAdsRequest or SearchGoogleAdsStreamRequest
                message class instance.

        Returns:
            A grpc.Call/grpc.Future instance representing a service response.
        """

        request_path = client_call_details.method
        request_method = request_path.split("/")[-1]

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
        
        message_serialized =  MessageToJson(request)

        request_serialized = {
            'request_class': request_class,
            'message_serialized': message_serialized
        }
    
        request_serialized_json = json.dumps(request_serialized)
        request_parsed = json.loads(request_serialized_json)

        request_class_obj = eval(request_parsed['request_class'])
        request_message_obj = request_parsed['message_serialized']

        request_deserialized = Parse(request_message_obj, request_class_obj())
   
        response = continuation(client_call_details, request_deserialized)
        
        if self.logger.isEnabledFor(logging.WARNING):
            self._log_request(client_call_details, request, response)

        return response

    def intercept_unary_stream(self, continuation, client_call_details,
                               request):

        """Intercepts and logs API interactions for Unary-Stream requests.

        Overrides abstract method defined in grpc.UnaryStreamClientInterceptor.

        Args:
            continuation: a function to continue the request process.
            client_call_details: a grpc._interceptor._ClientCallDetails
                instance containing request metadata.
            request: a SearchGoogleAdsRequest or SearchGoogleAdsStreamRequest
                message class instance.

        Returns:
            A grpc.Call/grpc.Future instance representing a service response.
        """
        def on_rpc_complete(response_future):
            if self.logger.isEnabledFor(logging.WARNING):
                self._log_request(client_call_details, request, response_future)
        print(client_call_details)
        response = continuation(client_call_details, request)

        response.add_done_callback(on_rpc_complete)

        return response
