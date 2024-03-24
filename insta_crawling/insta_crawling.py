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
import json
import re
from bs4 import BeautifulSoup
import unicodedata

try:
    with open('C:/Users/wogns/OneDrive/바탕 화면/Data_study/데이터사이언스/insta_crawling/config.json', 'r') as file:
        config = json.load(file)

except Exception as e:
    print("Error:", e)

ID = config['DEFAULT']['LOGIN_ID']
PW = config['DEFAULT']['LOGIN_PW']



# 웹 열기
dr = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
dr.get('https://www.instagram.com/') #인스타그램 웹 켜기
time.sleep(2) 	#2초 대기

# 로그인
id_box = dr.find_element(By.CSS_SELECTOR, "#loginForm > div > div:nth-child(1) > div > label > input")   #아이디 입력창
password_box = dr.find_element(By.CSS_SELECTOR, "#loginForm > div > div:nth-child(2) > div > label > input")     #비밀번호 입력창
login_button = dr.find_element(By.CSS_SELECTOR,'#loginForm > div > div:nth-child(3) > button')      #로그인 버튼

act = ActionChains(dr)      #동작 명령어 지정
act.send_keys_to_element(id_box, ID).send_keys_to_element(password_box, PW).click(login_button).perform() #아이디 입력, 비밀 번호 입력, 로그인 버튼 클릭 수행
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
time.sleep(5)


# 린가드 해시태그
dr.get('https://www.instagram.com/explore/tags/'+ '린가드')
search_list = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, '_aagw')))
search_list.click()

html = dr.page_source  # 페이지 소스 가져오기
soup = BeautifulSoup(html, 'html.parser')  # BeautifulSoup을 이용해 HTML 파싱

content = soup.select('div._a9zs > h1')[0].text
print(content)
time.sleep(5)
'''

try:
    # 1) 현재 페이지의 HTML 정보 가져오기
    html = dr.page_source
    soup = BeautifulSoup(html, 'lxml')


    # 2) HTML을 텍스트 파일로 저장하기
    with open('페이지.html', 'w', encoding='utf-8') as file:

        file.write(str(soup))


except Exception as e:
    print("오류 발생:", e)
    

try:
    content = soup.select('div._a9zs > h1')[0].text
    content = unicodedata.normalize('NFC',content)

except Exception as e:
    content = ''
    print(e)
    print(content)

     
tags = re.findall(r'#[^\s#,\\]+', content)


date = soup.select('._a9ze._a9zf')[0]['datetime'][:10]

try:
    like = soup.select('.x193iq5w.xeuugli.x1fj9vlw.x13faqbe.x1vvkbs.xt0psk2.x1i0vuye.xvs91rp.x1s688f.x5n08af.x10wh9bi.x1wdrske.x8viiok.x18hxmgj')[0].text
    like = re.findall(r'[\d]+', like)[0] 

except:
    print('')
    print(e)


    

data = [content, date, like, tags]


def move_next(driver):

    try:

        right = driver.find_element(By.CLASS_NAME,'_aaqh')

        right.click()

        time.sleep(3)

        

        # 다음 페이지가 완전히 로드될 때까지 기다리기 위해 클릭 후 대기를 추가합니다.

        # wait = WebDriverWait(driver, 10)

        # wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '다음_요소_선택자')))

        

    except Exception as e:

        print(f"오류가 발생했습니다: {e}")

​

move_next(driver)
'''