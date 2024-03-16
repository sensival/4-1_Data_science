from bs4 import BeautifulSoup
import requests

url = "https://www.mirna.kr/Main/Index.nm"
response = requests.get(url)

soup = BeautifulSoup(response.content, "html.parser")

selected_divs = soup.select_one("div.col-xs-3")

print(selected_divs)
