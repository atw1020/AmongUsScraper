"""

Author: Arthur wesley

"""

from http.server import BaseHTTPRequestHandler


class PostGameClassifierHandler(BaseHTTPRequestHandler):

    command = "POST"
    path = "/game-classifier"

    def handle(self):
        pass