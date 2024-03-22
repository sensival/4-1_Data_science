#작업에 필요한 라이브러리 
from selenium import webdriver 
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import time
from selenium.webdriver.common.by import By

#01. 웹 열기
dr = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
dr.get('https://www.instagram.com/') #인스타그램 웹 켜기
time.sleep(2) 	#2초 대기

#경로 지정
id_box = dr.find_element(By.CSS_SELECTOR, "#loginForm > div > div:nth-child(1) > div > label > input")   #아이디 입력창
password_box = dr.find_element(By.CSS_SELECTOR, "#loginForm > div > div:nth-child(2) > div > label > input")     #비밀번호 입력창
login_button = dr.find_element(By.CSS_SELECTOR,'#loginForm > div > div:nth-child(3) > button')      #로그인 버튼

#동작 제어
act = ActionChains(dr)      #동작 명령어 지정
act.send_keys_to_element(id_box, 'fever.study').send_keys_to_element(password_box, 'dlalstjs12').click(login_button).perform()     #아이디 입력, 비밀 번호 입력, 로그인 버튼 클릭 수행
time.sleep(2)
