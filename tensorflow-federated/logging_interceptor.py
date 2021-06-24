from grpc import UnaryUnaryClientInterceptor, UnaryStreamClientInterceptor
from interceptor import Interceptor
import logging
import json

class LoggingInterceptor(Interceptor, UnaryUnaryClientInterceptor,
                         UnaryStreamClientInterceptor):

    _FULL_REQUEST_LOG_LINE = ('Request\n-------\nMethod: {}\nHost: {}\n'
                              'Headers: {}\nRequest: {}\n\nResponse\n-------\n'
                              'Headers: {}\nResponse: {}\n')
    _FULL_FAULT_LOG_LINE = ('Request\n-------\nMethod: {}\nHost: {}\n'
                            'Headers: {}\nRequest: {}\n\nResponse\n-------\n'
                            'Headers: {}\nFault: {}\n')
    _SUMMARY_LOG_LINE = ('Request made: ClientCustomerId: {}, Host: {}, '
                         'Method: {}, RequestId: {}, IsFault: {}, '
                         'FaultMessage: {}')

    def __init__(self, logger, api_version, endpoint=None):
        """Initializer for the LoggingInterceptor.

        Args:
            logger: An instance of logging.Logger.
            api_version: a str of the API version of the request.
            endpoint: a str specifying the endpoint for requests.
        """
        super().__init__(api_version)
        self.endpoint = endpoint
        self.logger = logger

    def _get_trailing_metadata(self, response):
        """Retrieves trailing metadata from a response object.

        If the exception is a GoogleAdsException the trailing metadata will be
        on its error object, otherwise it will be on the response object.

        Returns:
            A tuple of metadatum representing response header key value pairs.

        Args:
            response: A grpc.Call/grpc.Future instance.
        """
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
        """Retrieves the initial metadata from client_call_details.

        Returns an empty tuple if metadata isn't present on the
        client_call_details object.

        Returns:
            A tuple of metadatum representing request header key value pairs.

        Args:
            client_call_details: An instance of grpc.ClientCallDetails.
        """
        return getattr(client_call_details, 'metadata', tuple())

    def _get_call_method(self, client_call_details):
        """Retrieves the call method from client_call_details.

        Returns None if the method is not present on the client_call_details
        object.

        Returns:
            A str with the call method or None if it isn't present.

        Args:
            client_call_details: An instance of grpc.ClientCallDetails.
        """
        return getattr(client_call_details, 'method', None)

    def _get_customer_id(self, request):
        """Retrieves the customer_id from the grpc request.

        Returns None if a customer_id is not present on the request object.

        Returns:
            A str with the customer id from the request or None if it isn't
            present.

        Args:
            request: An instance of a request proto message.
        """
        if hasattr(request, 'customer_id'):
            return getattr(request, 'customer_id')
        elif hasattr(request, 'resource_name'):
            resource_name = getattr(request, 'resource_name')
            segments = resource_name.split('/')
            if segments[0] == 'customers':
                return segments[1]
        else:
            return None

    def _parse_exception_to_str(self, exception):
        """Parses response exception object to str for logging.

        Returns:
            A str representing a exception from the API.

        Args:
            exception: A grpc.Call instance.
        """
        try:
            # If the exception is from the Google Ads API then the failure
            # attribute will be an instance of GoogleAdsFailure and can be
            # concatenated into a log string.
            return exception.failure
        except AttributeError:
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

    def _log_successful_request(self, method, customer_id, metadata_json,
                                request_id, request, trailing_metadata_json,
                                response):
        pass
        #print('Successful request: ')
        #print('Method: %s' % method)
        #print('Request: %s' % request)
        #print('Response: %s' % response.result())

    def _log_failed_request(self, method, customer_id, metadata_json,
                            request_id, request, trailing_metadata_json,
                            response):

        exception = self._get_error_from_response(response)
        exception_str = self._parse_exception_to_str(exception)
        fault_message = self._get_fault_message(exception)

        print('FAILED REQUEST')

    def _log_request(self, client_call_details, request, response):
        """Handles logging all requests.

        Args:
            client_call_details: An instance of grpc.ClientCallDetails.
            request: An instance of a request proto message.
            response: A grpc.Call/grpc.Future instance.
        """
        method = self._get_call_method(client_call_details)
        customer_id = self._get_customer_id(request)
        initial_metadata = self._get_initial_metadata(client_call_details)
        initial_metadata_json = self.parse_metadata_to_json(initial_metadata)
        trailing_metadata = self._get_trailing_metadata(response)
        request_id = self.get_request_id_from_metadata(trailing_metadata)
        trailing_metadata_json = self.parse_metadata_to_json(trailing_metadata)

        if response.exception():
            self._log_failed_request(
                method, customer_id, initial_metadata_json, request_id, request,
                trailing_metadata_json, response)
        else:
            self._log_successful_request(
                method, customer_id, initial_metadata_json, request_id, request,
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
        response = continuation(client_call_details, request)
        
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

        response = continuation(client_call_details, request)

        response.add_done_callback(on_rpc_complete)

        return response
