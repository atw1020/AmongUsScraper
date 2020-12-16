"""

Author: Arthur wesley

"""

from http.server import BaseHTTPRequestHandler


class PostGameClassifierHandler(BaseHTTPRequestHandler):

    command = "POST"
    path = "/game-classifier"

    def do_POST(self):
        """

        respond to a POST request

        :return:
        """
        pass
