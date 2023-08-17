import pandas as pd
import requests
from bs4 import BeautifulSoup
from html.parser import HTMLParser
import warnings
from urllib3.exceptions import InsecureRequestWarning
from multiprocessing import Pool, cpu_count

warnings.simplefilter('ignore', InsecureRequestWarning)


# HTML 태그 내부의 문자열을 추출하기 위한 HTMLParser 서브 클래스
class MyHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []

    def handle_data(self, data):
        self.text.append(data)


def process_url(idx, url):
    # 각 URL에서 HTML 추출
    try:
        response = requests.get(url, verify=False, timeout=5)
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while trying to scrape {url}: {str(e)}")
        return idx, '', '', 1  # 에러가 발생했을 때 빈 문자열과 카운트 1 반환

    print(response.text)
    # BeautifulSoup를 사용하여 HTML 파싱
    try:
        soup = BeautifulSoup(response.text, 'html.parser')

        # <style> 태그 제거
        for style in soup('style'):
            style.decompose()

        # HTML 문자열
        cleaned_html = str(soup)

        # HTML 태그 내부의 문자열을 추출
        parser = MyHTMLParser()
        parser.feed(cleaned_html)
        inner_text = ' '.join(parser.text)
    except Exception as e:
        print(f"An error occurred while parsing HTML from {url}: {str(e)}")
        return idx, '', '', 1  # 파싱 에러가 발생했을 때 빈 문자열과 카운트 1 반환

    return idx, cleaned_html, inner_text, 0  # 정상적으로 처리되었을 때 카운트 0 반환


if __name__ == '__main__':
    # DataFrame으로 CSV 파일 읽기
    df = pd.read_csv('../../url_data.csv')

    # 새로운 컬럼 'HTML'과 'Inner_Text' 생성
    df['HTML'] = ''
    df['Natural_Language'] = ''

    # URL 리스트 얻기
    urls = df['URL'].tolist()

    error_count = 0  # 에러 카운트 초기화

    with Pool(cpu_count()) as p:
        result = p.starmap(process_url, enumerate(urls))

    for idx, html, inner_text, error in result:
        try:
            df.at[idx, 'HTML'] = html
            df.at[idx, 'Natural Language'] = inner_text
            error_count += error  # 에러 카운트 업데이트
        except Exception as e:
            print(f"An error occurred while updating the result: {str(e)}")

    print(f"Total number of errors: {error_count}")  # 총 에러 수 출력

    # 결과를 원래 CSV 파일에 다시 쓰기
    df.to_csv('url_data.csv', index=False)
