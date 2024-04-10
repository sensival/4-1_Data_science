from bs4 import BeautifulSoup
import requests

url = "https://www.mirna.kr/Main/Index.nm"
response = requests.get(url)

soup = BeautifulSoup(response.content,"html.parser")

print(soup.find_all("h5"))
