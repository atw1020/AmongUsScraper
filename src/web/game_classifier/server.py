"""

author: Arthur Wesley

"""

import socketserver

from get_game_classifier_handler import GetGameClassifierHandler


def main():
    """

    main method

    :return:
    """

    try:

        port = 6931

        # instantiate the server
        server = socketserver.TCPServer(("", port), GetGameClassifierHandler)

        # print status
        print("Web server is running on port", port)

        # run the server
        server.serve_forever()

    except KeyboardInterrupt:

        print("Shutting down server...")

        # end the server
        server.socket.close()

        print("server successfully shut down")


if __name__ == "__main__":
    main()
