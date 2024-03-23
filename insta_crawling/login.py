from selenium import webdriver 
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common import NoSuchElementException,ElementNotInteractableException

# 웹 열기
dr = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
dr.get('https://www.instagram.com/') #인스타그램 웹 켜기
time.sleep(2) 	#2초 대기

# 로그인
id_box = dr.find_element(By.CSS_SELECTOR, "#loginForm > div > div:nth-child(1) > div > label > input")   #아이디 입력창
password_box = dr.find_element(By.CSS_SELECTOR, "#loginForm > div > div:nth-child(2) > div > label > input")     #비밀번호 입력창
login_button = dr.find_element(By.CSS_SELECTOR,'#loginForm > div > div:nth-child(3) > button')      #로그인 버튼

act = ActionChains(dr)      #동작 명령어 지정
act.send_keys_to_element(id_box, 'fever.study').send_keys_to_element(password_box, 'dlalstjs12').click(login_button).perform() #아이디 입력, 비밀 번호 입력, 로그인 버튼 클릭 수행
time.sleep(2)

#브라우저 꺼짐 방지 옵션
chrome_options = Options()
chrome_options.add_experimental_option("detach", True)

# 알림창 끄기
wait = WebDriverWait(dr, 10) # WebDriverWait 객체 생성
alert_01 = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, '_ac8f')))

alert_01.click()


alert_02 = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, '_a9_1')))

alert_02.click()
