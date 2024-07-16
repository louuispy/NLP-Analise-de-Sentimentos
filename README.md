# NLP-Analise-de-Sentimentos

Faaaaaaaaaaaaala galera, tudo bem com vocês? Espero que sim!  
  
Venho aqui compartilhar com vocês um projeto que eu fiz recentemente, que consiste basicamente em uma NLP para realizar a análise de sentimentos em comentários do X (antigo Twitter)! Foi um projeto que fiz um tempinho atrás, mas que eu ainda não havia compartilhado e documentado passo a passo do projeto aqui com vocês, então aqui está todo o projeto! 😊

---

## Construção do projeto

### 1. Preparando a base de dados

De início, foram realizadas as importações para as seguintes bibliotecas:
- `pandas`
- `string`
- `spacy`
- `random`
- `seaborn`
- `numpy`
- `re`
  
Agora vamos realmente começar o projeto!

Primeiro, utilizamos a biblioteca pandas para fazer a leitura da base de dados que será utilizada no treinamento da NLP, e imprimir na tela as 5 primeiras linhas do nosso DataFrame.
Para isso, utilizamos o seguinte código:

```python
base_treinamento = pd.read_csv('Train50.csv', on_bad_lines='skip', delimiter=';')
base_treinamento.head()
```
Com isso, temos a nossa base de dados atribuída à variável `base_treinamento`.

![image](https://github.com/user-attachments/assets/ccd6cd40-cb32-4199-bd93-4ce2958c7a8e)

Essa base de dados contém 50000 registros, que estão distribuídos em 5 colunas. Dessas 5 colunas, iremos utilizar apenas `tweet_text` e `sentiment`, que são respectivamente as colunas que contém os dados dos textos e suas respectivas emoções.

Para excluir as demais colunas que não serão utilizadas, digitamos:

```python
base_treinamento.drop(['id', 'tweet_date', 'query_used'], axis = 1, inplace = True)
```
Dessa forma, teremos o seguinte DataFrame:

![image](https://github.com/user-attachments/assets/d891101e-b77b-41de-aaf0-f22dcc4cabcd)

Com isso, podemos prosseguir para o pré-processamento dos textos!

---

### 2. Pré-processamento dos textos

Antes de treinarmos o nosso modelo, precisamos fazer um pré-processamento dos textos, uma vez que eles apresentam caracteres especiais, pontuações e afins. Iremos limpar os textos, e deixá-los apenas com palavras.

Para isso, vamos criar a nossa NLP, instânciando a biblioteca spaCy:
```python
pln = spacy.load('pt_core_news_sm')
pln
```

Além disso, vamos importar do spaCy o módulo STOP_WORDS.
Mas o que são STOP_WORDS? São basicamente palavras que possuem pouco significado, como preposições, artigos e etc.

```python
from spacy.lang.pt.stop_words import STOP_WORDS
stop_words = STOP_WORDS
```
Agora, vamos finalmente escrever a função que irá realizar todo o pré-processamento!

```python
def preprocessamento(texto):
    # Letras minúsculas
    texto = texto.lower()
    
    # Nome do usuário
    texto = re.sub(r"@[A-Za-z0-9$-_@.&+]+", ' ', texto)
    
    # URLs
    texto = re.sub(r"https?://[A-Za-z0-9./]+", ' ', texto)
    
    # Espaços em branco
    texto = re.sub(r" +", ' ', texto)
    
    # Emoticons
    lista_emocoes = {':)': 'emocaopositiva',
                    ':d': 'emocaopositiva',
                    ':(': 'emocaonegativa'}
    for emocao in lista_emocoes:
        texto = texto.replace(emocao, lista_emocoes[emocao])
        
    # Lematização
    documento = pln(texto)
    
    lista = []
    for token in documento:
        lista.append(token.lemma_)
        
    # Stop words e pontuações
    lista = [palavra for palavra in lista if palavra not in stop_words and palavra not in string.punctuation]
    lista = ' '.join([str(elemento) for elemento in lista if not elemento.isdigit()])
    
    return lista
```
Basicamente, neste pré-processamento, estamos retirando da nossa base de dados todas as letras maiúsculas, nomes de usuário, URLs e espaços em branco deixando apenas o mais puro texto. Além disso, substituímos os emoticons para suas respectivas emoções. Por fim, fazemos a lematização dos textos. A lematização o prcoesso de representar as palavras através do infinitivo dos verbos, ou seja, a gente vai praticamente reduzir a palavra até a sua raiz, e retirar todas as inflexões.

Após aplicarmos o pré-processamento utilizando o código:

```python
base_treinamento['tweet_text'] = base_treinamento['tweet_text'].apply(preprocessamento)
```

Temos o seguinte DataFrame:


![image](https://github.com/user-attachments/assets/75bf91e5-4612-4842-917f-d917665193b1)

Se você reparar, nós aplicamos o pré-processamento apenas na base de dados de treinamento. Agora, iremos aplicar na base de dados o teste.

```python
base_teste['tweet_text'] = base_teste['tweet_text'].apply(preprocessamento)
```
![image](https://github.com/user-attachments/assets/c667f84a-46f8-4764-ac83-604e31776a37)

---

### 3. Tratamento da classe

Agora vamos fazer o tratamento da classe de nossos textos. No DataFrame, os textos com emoções positivas são classificados como 1, e negativas como 0. Vamos realizar um tratamento que vai substituir esses valores por `POSITIVO` e `NEGATIVO`.

```python
base_dados_treinamento_final = []
for texto, emocao in zip(base_treinamento['tweet_text'], base_treinamento['sentiment']):
    if emocao == 1:
        dic = ({'POSITIVO': True, 'NEGATIVO': False})
    elif emocao == 0:
        dic = ({'POSITIVO': False, 'NEGATIVO': True})
    base_dados_treinamento_final.append([texto, dic.copy()])
```

Ao final do código, teremos uma lista, com os textos e suas emoções:

![image](https://github.com/user-attachments/assets/a8e86d78-4565-45d0-b394-15998c559b56)


---

### 4. Criação do Classificador

Vamos agora realizar o treinamento da nossa NLP!

De início, vamos digitar o seguinte código:

```python
from spacy.training import Example
modelo = spacy.blank('pt')
categorias = modelo.add_pipe("textcat")
categorias.add_label("POSITIVO")
categorias.add_label("NEGATIVO")
historico = []
```

No geral, nós criamos uma variável modelo que realiza uma instância do spaCy, e dessa variável, adicionamos labels 'POSITIVO' e 'NEGATIVO'.
Também criamos uma lista vazia, na variável histórico, que iremos utilizar logo mais.

Agora, vamos para a parte mais "difícil" do projeto:

```python
modelo.begin_training()
for epoca in range(5):
    random.shuffle(base_dados_treinamento_final)
    losses = {}
    for batch in spacy.util.minibatch(base_dados_treinamento_final, 512):
        textos = [modelo(texto) for texto, entities in batch]
        annotations = [{'cats': entities} for texto, entities in batch]
        examples = [Example.from_dict(doc, annotation) for doc, annotation in zip(textos, annotations)]
        modelo.update(examples, losses=losses)
        historico.append(losses)
    if epoca % 5 == 0:
        print(losses)
```
Essa parte pode parecer bem complexa, mas vamos explicar! 😅
A primeira linha do código vai indicar o início do treinamento e prepara o modelo em si para receber os dados de treino e as atualizações dos pesos. 

Logo em seguida, criamos um laço de repetição em um `range(5)`, pois iremos treinar nosso modelo em 5 épocas. 

Em seguida, temos o código `random.shuffle(base_dados_treinamento_final)`, que vai basicamente embaralhar tudo na base de dados de treinamento, evitando assim que o modelo aprenda a ordem dos dados, e apresente resultados com baixa eficácia nos testes. 

Após isso, criamos uma variável losses, que recebe um dict. É nessa variável que vamos armazenar os erros do modelo durante o treinamento.

Em seguida, no código `for batch in spacy.util.minibatch(base_dados_treinamento_final, 512):`, nós dividimos os dados de treinamento em 512 pedaços, ou lotes, ou batches, que seria o nome correto. Nós faemos para que o modelo realize o pré-processamento em um pedaço de cada vez. Isso torna o treinamento do modelo mais rápido e bem mais eficiente, na prática.

No trecho seguinte, o `modelo` será aplicado a cada texto presente no batch. Ou seja, será feito um pré-processamento nos textos do batch para criar objetos de documento `doc`.

Logo em seguida, criamos uma lista, e dentro dessa lista temos um dict, que irá associar a chave `cats` às categorias presentes em cada texto no batch.

No trecho seguinte, nós criamos exemplos de treinamento, utilizando os textos já pré-processados. Isso é feito a partir de um `doc` e sua respectiva anotação. A função `Example.from_dict` serve para converter esses documentos e anotações em um formato compatível com o treinamento do modelo.

Depois disso, chamamos o método `update` do modelo para atualizar os pesos do modelo com base nos exemplos fornecidos no trecho anterior, e fazer um cálculo das perdas, que será armazenados em `losses`.

Logo em seguida, nós iremos fazer um `append` dos erros na lista `histórico`.

Por fim, após as 5 épocas, imprimos na tela o dicionário de perdas, apra assim, monitorarmos o progresso do treinamento da NLP.

---

E é isso gente, concluímos nossa NLP! Espero que tenham gostado! 🚀
Caso vocês queiram aprender mais sobre, recomendo muito os cursos do Jonas Granatyr, da Udemy! ^^
