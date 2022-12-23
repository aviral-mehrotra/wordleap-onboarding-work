import requests
import os

BASE = 'http://127.0.0.1:8080/'

response = requests.post(BASE + "wordInContextTranslation", {"src": "Learn how you can put your company's SRC membership to work for you today.", "tgt" : "了解如何让贵公司的 SRC 会员资格为您服务。", "index" : 0})
print(response.json())