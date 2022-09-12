from prometheus_client import start_http_server

class PrometheusClient():
    
    def __init__(self, port=9103) -> None:
        start_http_server(port)