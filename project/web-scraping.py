import requests
from bs4 import BeautifulSoup

offset = 2100
max_offset = 3000
offset_increment = 12

BASE_URL = 'http://www.forexrate.co.uk/'

news_archive = []

for i in range(offset, max_offset, offset_increment):
    
  url = f'http://www.forexrate.co.uk/newsarchive.php?start={i}'
  print(url)
  
  page = requests.get(url)
  soup = BeautifulSoup(page.content, 'html.parser')
  tables = soup.findChildren('table')
  news_table = tables[1]
  rows = news_table.findChildren(['th', 'tr'])
  
  for idx,row in enumerate(rows):

    if idx == 0:
        continue
    cells = row.findChildren('td')
    
    for idx,cell in enumerate(cells):
      txt = cell.text
      href = cell.find('a')['href']
      href = BASE_URL + href.replace('./','')

      if "newsarchive.php?start=" in href:
        continue

      # let's get the date of the article
      date_page = requests.get(href)
      date_soup = BeautifulSoup(date_page.content, 'html.parser')
      date_div = date_soup.findChildren('div')[3]
      date_str = date_div.text
      news_archive.append({'txt':txt,'url':href,'date':date_str})
      print(len(news_archive), date_str, {'txt':txt,'url':href,'date':date_str})

import pandas as pd
news_archive_df = pd.DataFrame(news_archive)
news_archive_df.to_csv('news.csv')