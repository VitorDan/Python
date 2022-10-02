from settings import DevConvfig as DevCfg
import pandas as pd
file_name = DevCfg['FILEPATH']
df =  pd.read_csv(file_name,sep=',')
print(df.describe())
print(f'Dados da coluna Multiplicador possuem media de {df["Multiplicador"].mean()} com desvio padrão de {df["Multiplicador"].std()}')
df.loc[df['Multiplicador']>100.01,'Multipliacador'] = 100.01 #localiza valores maiores que 100.01 e converge os para 100.1
resultados = df.loc[df['Dia'].isin([4,11,18,25,5,12,19,26]) ]# encontra valores do dataset que corresponde aos dias da lista
resultados.to_csv('fins_de_semana.csv',sep=',', index=False)# gera um csv com os resultados