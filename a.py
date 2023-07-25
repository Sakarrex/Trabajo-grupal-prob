from bs4 import BeautifulSoup
import requests
import pandas as pd
url = "http://wiki.stat.ucla.edu/socr/index.php/SOCR_Data_MLB_HeightsWeights"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
rows = soup.find('center').find('table').find_all('tr')
del rows[0]
lista = []
for row in rows:
    dict = {}
    dict['Name'] = row.find_all('td')[0].text
    dict['Team'] = row.find_all('td')[1].text
    dict['Position'] = row.find_all('td')[2].text
    dict['Height'] = row.find_all('td')[3].text
    dict['Weight'] = row.find_all('td')[4].text
    dict['Age'] = float(row.find_all('td')[5].text)

    lista.append(dict)

dt = pd.DataFrame(lista)
dt.to_csv("players.csv",index=False)