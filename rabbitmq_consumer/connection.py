import pika
import logging
import time
from shared.log import custom_logger


class RabbitMQConnection:
    def __init__(self, amqp_url):
        self.amqp_url = amqp_url
        self.connection = None
        self.channel = None
        self.connect()

    def connect(self):
        """
        Establish a connection to RabbitMQ.
        """
        try:
            params = pika.URLParameters(self.amqp_url)
            self.connection = pika.BlockingConnection(params)
            self.channel = self.connection.channel()
        except pika.exceptions.AMQPConnectionError as e:
            custom_logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise

    def reconnect(self, retry_interval=5):
        """
        Attempt to reconnect to RabbitMQ with a specified interval.
        """
        while True:
            try:
                self.connect()
                return
            except pika.exceptions.AMQPConnectionError:
                custom_logger.error(
                    f"Connection to RabbitMQ failed. Retrying in {retry_interval} seconds."
                )
                time.sleep(retry_interval)
