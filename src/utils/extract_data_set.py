import csv
from google_trends import daily_trends, realtime_trends
import pandas as pd
from datetime import datetime, timedelta
from googlesearch import search
from pytrends.request import TrendReq
import random
from itertools import permutations

keywords = []
filename = '../../data/raw/related_queries_csvs/relatedQueries_0706.csv'
with open(filename, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # skip first row
    next(reader)  # skip
    for row in reader:
        if not row:
            continue
        if row[0] == "RISING":
            break
        keyword = row[0]
        keywords.append(keyword)

# print(keywords)
random.shuffle(keywords)
p_li = list(permutations(keywords, 2))
random.shuffle(p_li)

random.shuffle(keywords)
p_li2 = list(permutations(keywords, 2))
random.shuffle(p_li2)
hap = []
for (a1, a2), (b1, b2) in zip(p_li, p_li2):
    c1 = a1 + " " + b1
    c2 = a2 + " " + b2
    hap.append((c1, c2))
url_list = []
for i in hap:
    if len(url_list) > 100:
        break
    j = 0
    for result in search(i[0] + " " + i[1]):
        if (j == 5):
            break
        if (result.find("instagram.com") == -1 and result.find("melon.com") == -1 and result.find(
                "namu") == -1 and result.find("naver.com") == -1 and result.find("go.kr") == -1 and result.find(
            "co.kr") == -1 and result.find("youtube.com") == -1):
            url_list.append((i[0] + " " + i[1], result, 0))  # 라벨링 값으로 0을 넣음
            j = j + 1

# CSV 파일 경로
csv_output = "result_csvs/url_data.csv"

# CSV 파일에 저장할 데이터
data = [("Keyword", "URL", "Label")] + url_list

# CSV 파일로 데이터 작성
with open(csv_output, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)

print(f"CSV 파일 '{csv_output}'이 저장되었습니다.")