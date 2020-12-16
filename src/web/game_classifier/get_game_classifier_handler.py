"""

Author: Arthur wesley

"""

from http.server import BaseHTTPRequestHandler


class GetGameClassifierHandler(BaseHTTPRequestHandler):

    command = "GET"
    path = "/game-classifier"

