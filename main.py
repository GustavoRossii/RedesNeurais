import pandas as pd

tabela = pd.read_csv('filmes_disney.csv')
#print(tabela)
print('-'*30)

tabela = tabela.sort_values(by='arrecadacao', ascending=False )
print('-'*30)

top3 = tabela.head(3)
print(top3)
print('-'*30)

print(top3[['titulo', 'arrecadacao']])
print('-'*30)

num = int(input('Qts filmes:'))
topFilmes = tabela.head(num)

print(f'TOP {num} FILMES')
print('-'*30)
print(topFilmes[['titulo','arrecadacao']])
print('-'*30)