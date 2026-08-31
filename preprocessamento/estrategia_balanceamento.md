**1. Visão geral e contexto**

Projeto desenvolvido na Faculdade Donaduzzi para o curso de Inteligência Artificial. 
O repositório faz parte da disciplina de Projeto integrador 
que visa análises e classificação de nódulos pulmonares em exames de tomografia computadorizada.

O conjunto de dados possui um desbalanceamento natural acentuado entre os tipos de tumores,
sendo que há uma quantidade consideravelmente maior de amostras benignas do que malignas. O objetivo dessa etapa 
é aplicar estratégias de balanceamento para que o modelo de machine learning entenda a importância das classes 
durante o treinamento, sem viés de classe majoritária.

**2. Dependências e Bibliotecas Utilizadas**

O projeto foi inteiramente realizado utilizando do Python. Para tanto, A estrutura de módulos é dividida entre dependências 
nativas e bibliotecas externas.

_Nativas:_ 

Para utilização dessas é necessário apenas rodar os códigos todos os imports necessários já foram feitos

sys: Gerencia o encerramento do script e a comunicação direta com o interpretador em situações de erro;

argparse: Realiza a leitura e validação dos argumentos passados via linha de comando no momento do disparo do script;

pathlib: Manipula caminhos de arquivos e diretórios de forma a ser compatível com Windows, Linux e macOS. 
Dessa biblioteca foi importado o path;

__future__: Suporte interno do Python para o uso de dicas de tipagem moderna sem quebrar a execução em versões mais antigas do ambiente. 
Desta foi importada o recurso annotations;

_Externas_

Para realizar a execução dessa biblioteca é necessário fazer a instalação da mesma. Nesse caso, faça o pip install

pandas: Faz a manipulação de dados e a leitura de arquivos em CSV;

**3. Estratégia Escolhida**

_Pesos de Classe na Função de Perda (Loss Function):_ Em vez de igualar o número de amostras à força, ajusta-se a função de custo 
da rede neural. Atribuem-se pesos maiores para os erros cometidos na classe minoritária (tumores malignos), penalizando o modelo 
com mais severidade quando ele erra a identificação dessa classe.

_Data augmentation:_ Para diversificar a base de treino, são aplicadas transformações nas imagens de tomografia:

Rotação aleatória de até 15 graus;

Espelhamento horizontal e vertical;

Deslocamento central (jitter) de até 3 mm no centro do nódulo;

**4. Sobre o SMOTE**

Optamos por não utilizar o algoritmo SMOTE tendo em vista que nele é feita a criação de novos dados fazendo a interpolação de características
entre dois exemplos. Embora seja funcional em dados vetoriais simples, em exames de tomografia computadorizada é gerado uma combinação
artificial de pixels inexistentes na anatomia real de um pulmão.

**5. Isolamento dos conjuntos**

Para fazermos as análises dos nódulos, separamos-os em três categorias. São elas: treinamento, validação e teste. De um total de 100%
Dividimos respectivamente 70, 15 e 15 por cento nessas três categorias. As estratégias de balanceamento dos dados foi realizada apenas 
na etapa de treinamento. Mantendo assim, as etapas de validação e teste intocadas de modo à manter a proporção e distribuição real dos casos clínicos.



artificial de pixels inexistentes na anatomia real de um pulmão.

**5. Isolamento dos conjuntos**

Para fazermos as análises dos nódulos, separamo-os em três categorias. São elas: treinamento, vaçidação e teste. De um total de 100%
dividimos respectivamente 70, 15 e 15 porcento nessas três categoria. As estratégias de balanceamento dos dados foi realizada apenas 
na etapa de treinamento. Mantendo assim, as etapas de validação e teste intocadas de modo à manter a proporção e distribuição real dos casos clínicos.



