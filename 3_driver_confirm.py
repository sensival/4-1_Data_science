'''
## selenium, webdriver 설치 확인
import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

driver = webdriver.Chrome('"C:\tools\chromedriver.exe"') ## 크롬 드라이버가 위치한 경로 대입 필요
'''

'''

from selenium import webdriver 
from webdriver_manager.chrome import ChromeDriverManager

# Chrome WebDriver 초기화
driver = webdriver.Chrome(executable_path=ChromeDriverManager().install())
'''

from selenium import webdriver 
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

#웹드라이버 설정
driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
