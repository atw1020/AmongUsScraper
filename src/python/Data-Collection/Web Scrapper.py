"""

Author: Arthur wesley, Gregory Ghiroli

https://medium.com/swlh/scraping-live-stream-video-with-python-b7154b1fedde

"""

from src.python import constants

from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities


def init_webdriver():
    """

    initializes the selenium webdriver

    :return:
    """

    dimensions = str(constants.dimensions[0]) + ", " + str(constants.dimensions[1])

    caps = DesiredCapabilities.CHROME
    caps["goog:loggingPrefs"] = {"performance": "ALL"}

    options = webdriver.ChromeOptions()
    options.add_argument("--window-size=" + dimensions)
    options.add_argument("headless")

    driver = webdriver.Chrome(desired_capabilities=caps, options=options)
    driver.get("twitch.tv")

    return driver


def main():
    """

    main method

    :return: None
    """

    driver = init_webdriver()


if __name__ == "__main__":
    main()
