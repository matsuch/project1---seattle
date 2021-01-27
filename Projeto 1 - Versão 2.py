#Importando algebra linear e manipulação de dados
import numpy as np
import pandas as pd

#Pacotes de plotagem de gráficos
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

#Pacotes de Machine Learning
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#Abrindo os banco de dados

df_review = pd.read_csv('./reviews.csv')
df_listings = pd.read_csv('./listings.csv')
df_calendar = pd.read_csv('./calendar.csv')


#==============================================================================
#Separa as colunas conforme o tipo de informação - Host, Review, quarto, bairro
#==============================================================================

host = df_listings[['host_is_superhost', 'host_response_rate', 'host_response_time']]

review = df_listings[['number_of_reviews', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
          'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value']]

quarto = df_listings[['room_type', 'bathrooms', 'bedrooms', 'beds', 'bed_type', 'accommodates','property_type']]

bairro = df_listings[['neighbourhood', 'neighbourhood_cleansed','neighbourhood_group_cleansed']]

#==============================================================
#Verifica quais informações são itens em cada um dos novos df's
#==============================================================

#Contar vazios em cada coluna

#host
print(host.isnull().sum()) #Contar nro de vazios por colunas
#Transforma a coluna de response_rate em str e depois em float para achar a média
host['host_response_rate_num'] = host['host_response_rate'].astype(str)
host['host_response_rate_num'] = host['host_response_rate_num'].str.replace("%", "").astype("float")
host['host_response_rate_num'].fillna(host['host_response_rate_num'].mean(), inplace = True)

#Trata os dados de response_time e transforma em float para preencher pela media
host['host_response_time'] = host['host_response_time'].map({'within a few hours':6,
                                                           'within an hour':1,
                                                           'within a day':24,
                                                           'a few days or more':48})
host['host_response_rate_num'].fillna(host['host_response_rate_num'].mean(), inplace = True)
host['host_response_time'].fillna(host['host_response_time'].mean(), inplace = True)

host.drop(['host_response_rate'], axis=1)
hostnew = host.dropna()

#review
print(review.isnull().sum()) #Contar nro de vazios por colunas
#Preencher todos os dados de review pela média, já que todos são valores números que fazem sentido ser a media
review['review_scores_value'].fillna(review['review_scores_value'].mean(), inplace = True)
review['review_scores_rating'].fillna(review['review_scores_rating'].mean(), inplace = True)
review['review_scores_accuracy'].fillna(review['review_scores_accuracy'].mean(), inplace = True)
review['review_scores_cleanliness'].fillna(review['review_scores_cleanliness'].mean(), inplace = True)
review['review_scores_checkin'].fillna(review['review_scores_checkin'].mean(), inplace = True)
review['review_scores_communication'].fillna(review['review_scores_communication'].mean(), inplace = True)
review['review_scores_location'].fillna(review['review_scores_location'].mean(), inplace = True)

#quarto
print(quarto.isnull().sum()) #Contar nro de vazios por colunas
quartonew=quarto.dropna() #Tira todos os vazios, pois são poucos dados

#bairro
print(quarto.isnull().sum()) #Contar nro de vazios por colunas
bairronew=bairro.dropna() #Tira todos os vazios, pois são poucos dados (10%) - Analises precisam do bairro

#Cria um novo banco de dados com as colunas selecionadas
df_novo = pd.concat((review, quartonew, hostnew, bairronew), axis=1)

#Separando as informações de data em mês e ano

df_calendar['date'] = pd.to_datetime(df_calendar['date'], format= '%Y/%m/%d') #colocando a data em datetime
df_calendar['ano'] = df_calendar['date'].dt.year #criando uma coluna de ano baseado na data
df_calendar['mês'] = df_calendar['date'].dt.month #criando uma coluna de mês baseado na data
df_calendar.drop(['date'], axis=1, inplace=True) #retirando a coluna 'Date' do banco de dados

#====================
#Tratamento dos dados
#====================

df_novo['host_is_superhost'] = df_novo['host_is_superhost'].map({'f':0,'t':1}) #Trocando os termos de f (false) e t (true) para 1 e 0

df_novo = pd.concat((df_novo, df_listings['id']), axis=1) #incluindo o ID único no novo banco de dados

df_novo.rename(index=str, columns={'id': 'listing_id'}, inplace=True) #trocando o nome da coluna id para listing_id

df = pd.merge(df_calendar, df_novo, on='listing_id') #unindo os dois banco de dados, baseado na listing_id

df.dropna(subset=['price'],inplace=True) #excluindo todas as linhas das acomodacões que não possuem preço listado.

df['price'] = df['price'].str.replace("[$, ]", "").astype("float") #convertendo 'price' em float (retirando o cifrão)
'''
#========================================================================
#Pergunta 1 - Como se comportam os preços das hospedagens?
#========================================================================

df_hist=df.groupby('listing_id')['price'].mean()
plt.figure(figsize=(15,10))
plt.hist(df_hist,bins=100)
plt.ylabel('Contagem')
plt.xlabel('Preço das hospedagens, USD')
plt.title('Histograma dos preços das hospedagens')
plt.savefig('Histograma de preços.png')

#========================================================================
#Pergunta 2 - Quais são os melhores periodos do ano para visitar Seattle?
#========================================================================

#Laço que percorre a coluna de mês e soma quantas vezes aquele mês apareceu
hospedagens_mensais = pd.Series([12])
for i in range(1, 13):
    hospedagens_mensais[i] = len(df[(df['mês'] == i) & (df['ano'] == 2016)]['listing_id'].unique())

hospedagens_mensais = hospedagens_mensais.drop(0) #excluo a contagem que aparece com 0 (12x)

meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai' , 'Jun' , 'Jul' , 'Ago' , 'Set' , 'Out' , 'Nov' , 'Dez']
preco_mensal = df.groupby('mês')['price'].mean() #agrupo o preço pelo mês e encontro a média de preço mensal

#plota o gráfico - Preço x Locacões - por mês
preco_mensal = df.groupby('mês')['price'].mean()
plt.subplots(figsize = (15,10))
ax = plt.gca()
sns.pointplot(x = meses, y = hospedagens_mensais, color='black',linestyles=['-'])
ax2=plt.twinx()
sns.pointplot(x = meses, y = preco_mensal, color='black', linestyles=['--'])
plt.legend(labels=['Preço Mensal', 'Nro Hospedagens disponiveis'])
plt.title('Hospedagens disponiveis, 2016')
plt.ylabel('Preço médio')
plt.xlabel('Meses')
plt.savefig('Periodo de locações mensais x preço.png')

#=============================================
#Pergunta 3 - Quais são os bairros mais caros?
#=============================================

preco_bairros = df.groupby('neighbourhood')['price'].mean().sort_values(ascending=False).reset_index(name='price')

top5_bairros_caros_nome = preco_bairros['neighbourhood'][0:5].tolist()
top5_bairros_caros_preco = preco_bairros['price'][0:5].tolist()   
top5_bairros_caros_qtd = [df.loc[df.neighbourhood == preco_bairros['neighbourhood'][0]  , 'neighbourhood'].count(),
                          df.loc[df.neighbourhood == preco_bairros['neighbourhood'][1]  , 'neighbourhood'].count(),
                          df.loc[df.neighbourhood == preco_bairros['neighbourhood'][2]  , 'neighbourhood'].count(),
                          df.loc[df.neighbourhood == preco_bairros['neighbourhood'][3]  , 'neighbourhood'].count(),
                          df.loc[df.neighbourhood == preco_bairros['neighbourhood'][4]  , 'neighbourhood'].count()]
                         
df_top5 = pd.DataFrame(list(zip(top5_bairros_caros_nome,top5_bairros_caros_qtd,top5_bairros_caros_preco)),columns=['Bairro','QTD','Preço'])


df_top5 = df_top5.set_index('Bairro')
fig = plt.figure(figsize=(10,10)) # Create matplotlib figure

ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as a
width = .3

df_top5.QTD.plot(kind='bar',color='red',ax=ax,width=width, position=0,legend=True)
df_top5.Preço.plot(kind='bar',color='blue', ax=ax2,width = width,position=1,legend=True)

teste = ['QTD','Preço']

ax.set_ylabel('QTD')
ax2.set_ylabel('Preço')
ax.legend(loc='upper center')
fig.autofmt_xdate(rotation=45)

ax.set_xlim(-1,5)
fig.savefig('Top 5 - Bairros mais caros - Preço x Disponibilidade.png')

#=================================================
#Pergunta 3.1 - Quais são os bairros mais baratos?
#=================================================

preco_bairros = df.groupby('neighbourhood')['price'].mean().sort_values(ascending=False).reset_index(name='price')

tamanho = len(preco_bairros.index)
print(tamanho)
top5_bairros_caros_nome = preco_bairros['neighbourhood'][76:81].tolist()
top5_bairros_caros_preco = preco_bairros['price'][76:81].tolist()   
top5_bairros_caros_qtd = [df.loc[df.neighbourhood == preco_bairros['neighbourhood'][tamanho-1]  , 'neighbourhood'].count(),
                          df.loc[df.neighbourhood == preco_bairros['neighbourhood'][tamanho-2]  , 'neighbourhood'].count(),
                          df.loc[df.neighbourhood == preco_bairros['neighbourhood'][tamanho-3]  , 'neighbourhood'].count(),
                          df.loc[df.neighbourhood == preco_bairros['neighbourhood'][tamanho-4]  , 'neighbourhood'].count(),
                          df.loc[df.neighbourhood == preco_bairros['neighbourhood'][tamanho-5]  , 'neighbourhood'].count()]
                         
df_top5 = pd.DataFrame(list(zip(top5_bairros_caros_nome,top5_bairros_caros_qtd,top5_bairros_caros_preco)),columns=['Bairro','QTD','Preço'])


df_top5 = df_top5.set_index('Bairro')
fig = plt.figure(figsize=(10,10)) # Create matplotlib figure

ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as a
width = .3

df_top5.QTD.plot(kind='bar',color='red',ax=ax,width=width, position=0,legend=True)
df_top5.Preço.plot(kind='bar',color='blue', ax=ax2,width = width,position=1,legend=True)

teste = ['QTD','Preço']

ax.set_ylabel('QTD')
ax2.set_ylabel('Preço')
ax.legend(loc='upper center')
fig.autofmt_xdate(rotation=45)

ax.set_xlim(-1,5)
fig.savefig('Top 5 - Bairros mais baratos - Preço x Disponibilidade.png')

#============================================================
#Pergunta 4 - Quais são os fatores que mais impacto no preço?
#============================================================

#Dados de preços das acomodações
preco_minimo = df['price'].max()
preco_maximo = df['price'].min()
preco_medio = df['price'].mean()

print('Hospedagem mais cara: USD ', preco_maximo)
print('Hospedagem mais barata: USD ', preco_minimo)
print('Media de preço das hospedagens: USD ', preco_medio)

#heatmap geral
df_hc = df.select_dtypes(include=['float','int64']).copy()
fig, ax = plt.subplots(figsize=(20,10))
sns.heatmap(df_hc.corr(), annot=True, fmt='.2f',ax=ax)
plt.savefig('Heatmap - Geral.png')

#Heatmap apenas dos comodos
df2 = df.copy()
df_fisico = df2[['bathrooms', 'bedrooms', 'beds', 'accommodates','price']]
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(df_fisico.corr(), annot=True, fmt='.2f',ax=ax)
plt.savefig('Heatmap - Espaço Fisico.png')

#Heatmap apenas das reviews
df_review = df2[['review_scores_rating','review_scores_accuracy', 'review_scores_cleanliness','review_scores_checkin',
                 'review_scores_communication', 'host_response_rate', 'host_response_time', 'host_response_rate_num' ,
                 'review_scores_value']]
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(df_review.corr(), annot=True, fmt='.2f',ax=ax)
plt.savefig('Heatmap - Review.png')
'''
#==============================================
#PROCESSO DE MACHINE LEARNING - 
#==============================================

#turn categorical columns into dummies

cat_col = list(df.select_dtypes(include=['object']).columns)

def create_dummy_df(df, cat_cols, dummy_na):

    for col in cat_cols:
            df = pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col], prefix=col, prefix_sep='_', drop_first=True, dummy_na=dummy_na)], axis=1)
    return df

cat_df = create_dummy_df(df, cat_col, dummy_na=False)

#Processo de linearização

#1. Drop the rows with missing response values
cat_df = cat_df.dropna(subset=['price'])

#2. Drop columns with Nan for all the values

df = df.dropna()

#3 Apply dummy_df
df = create_dummy_df(df,cat_col,dummy_na=True)

#4 Split data into X matriz and response vector y
X = df.drop(['price'], axis=1)
y = df['price']

#5 Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=42)

lm_model = LinearRegression(normalize=True) # Instantiate
lm_model.fit(X_train, y_train) #Fit

#Predict using your model
y_test_preds = lm_model.predict(X_test)
y_train_preds = lm_model.predict(X_train)

#Score using your model
test_score = r2_score(y_test, y_test_preds)
train_score = r2_score(y_train, y_train_preds)

#Print training and testing score
print("The rsquared on the training data was {}.  The rsquared on the test data was {}.".format(train_score, test_score))




