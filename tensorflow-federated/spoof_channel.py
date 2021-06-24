
class SpoofChannel(_channel.Channel):

  def __init__(self, ip_address, port):

    target = '{}:{}'.format(ip_address, port)

    super(_channel.Channel, self).__init__()
    self.ip_address = ip_address
    self.port = port
    
    self.channel = grpc.insecure_channel('{}:{}'.format(ip_address, port))

  def __enter__(self):
    print('enter')
    return self.channel.__enter__(self)
  
  def close(self):
    print('close')
    return self.channel.close()
  
  def subscribe(self, callback, try_to_connect=False):
    print('sub')
    def wrapped_callback(*args, **kwargs):
      print('Subscribe: channel status: %s' % args[0])
      callback(*args, **kwargs)
    return self.channel.subscribe(wrapped_callback, try_to_connect=try_to_connect)

  def unsubscribe(self, callback):
    print('unsub')
    return self.channel.unsubscribe(callback)

  def unary_unary(self, method, request_serializer=None, response_deserializer=None):
    print('unary_unary')
    return self.channel.unary_unary(method, request_serializer=request_serializer, 
      response_deserializer=response_deserializer)

  def stream_stream(self, method, request_serializer=None, response_deserializer=None):
    print('stream_stream')
    return self.channel.stream_stream(method, request_serializer=request_serializer, 
      response_deserializer=response_deserializer)
  
  def stream_unary(self, method, request_serializer=None, response_deserializer=None):
    print('stream_unary')
    return self.channel.stream_unary(method, request_serializer=request_serializer, 
      response_deserializer=response_deserializer)
  
  def unary_stream(self, method, request_serializer=None, response_deserializer=None):
    print('unary_stream')
    return self.channel.unary_stream(method, request_serializer=request_serializer, 
      response_deserializer=response_deserializer)
