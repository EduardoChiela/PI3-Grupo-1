# Conferência dos números da seleção de nódulos

- **Responsável pela conferência:** Rafael Casarin
- **Data da conferência:** 30/08/2026
- **Fonte utilizada:** banco de anotações do LIDC-IDRI distribuído com `pylidc` 0.2.3

## Resumo das checagens

| Checagem | Resultado observado | Situação |
|---|---:|---|
| Total de exames no `pylidc` | 1.018 | Conforme |
| Nódulos com consenso de 3 ou mais radiologistas | 1.192 | Próximo da referência de 1.186 |
| Fechamento das somas em cada filtro | Todas as somas fecham | Conforme |
| Proporção entre benignos e malignos | 51,66% / 48,34% | Fora da faixa esperada |

## 1. Total de exames

A consulta `pl.query(pl.Scan).count()` devolveu **1.018 exames**, que é o total esperado para o LIDC-IDRI. Portanto, não há indicação de banco de anotações incompleto.

**Resultado:** conforme.

## 2. Comparação com a referência do LUNA16

Depois do filtro de espessura de corte menor ou igual a 2,5 mm, permaneceram **897 exames**. O agrupamento das anotações gerou **2.291 candidatos a nódulo**, dos quais **1.192** tinham consenso de pelo menos três radiologistas.

A referência informada para comparação é de **1.186 nódulos**. O resultado obtido ficou 6 nódulos acima da referência, uma diferença de aproximadamente **0,51%**:

`(1.192 - 1.186) / 1.186 × 100 = 0,51%`

Essa diferença é pequena e o total pode ser considerado próximo da referência. Ainda assim, ela deve ser mantida registrada, pois os critérios deste projeto não reproduzem necessariamente todos os critérios de exclusão usados pelo LUNA16.

**Resultado:** conforme, com diferença registrada.

## 3. Fechamento do funil

As somas foram conferidas em todos os degraus:

- Filtro por espessura: `1.018 = 897 incluídos + 121 excluídos`.
- Filtro por consenso: `2.291 = 1.192 com consenso ≥ 3 + 1.099 com consenso < 3`.
- Classificação pela mediana: `1.192 = 388 benignos + 441 indeterminados + 363 malignos`.
- Conjunto final rotulado: `751 = 388 benignos + 363 malignos`.

Os **441 nódulos com mediana exatamente igual a 3** foram excluídos por indeterminação. Assim, o conjunto final contém **751 nódulos com rótulo binário**.

**Resultado:** conforme.

## 4. Proporção entre as classes

A proporção foi calculada somente sobre os **751 nódulos que receberam rótulo binário**, sem contar os casos indeterminados:

- Benignos: `388 / 751 = 51,66%`.
- Malignos: `363 / 751 = 48,34%`.

A proporção observada ficou fora da faixa de referência indicada, de 55/45 a 65/35 entre benignos e malignos. A diferença não deve ser corrigida alterando rótulos ou removendo casos apenas para fazer a distribuição entrar na faixa. O arquivo `nodulos_selecionados.csv`, gerado pelo script `selecao_nodulos.py`, foi conferido e aplica os critérios definidos: espessura menor ou igual a 2,5 mm, consenso de pelo menos três radiologistas, mediana dos escores e exclusão da mediana 3.

**Resultado:** não conforme com a faixa esperada; requer comparação com o CSV final.

## Observação sobre a conferência

Após a execução do script `selecao_nodulos.py`, o arquivo `nodulos_selecionados.csv` foi regravado e conferido. Ele contém exatamente **751 linhas**, com **388 benignos** e **363 malignos**, mantendo o fechamento do funil descrito neste documento.

Também foram encontrados nódulos finais com mais de quatro anotações no CSV: 6 linhas com 5 anotações, 2 linhas com 6 anotações e 1 linha com 7 anotações. Como o LIDC-IDRI possui até quatro radiologistas por exame, esses agrupamentos devem ser revisados no script de seleção ou registrados como uma limitação do agrupamento automático.

## Conclusão

O banco contém os 1.018 exames esperados, o total com consenso ficou muito próximo da referência do LUNA16 e todas as somas do funil fecharam. A única checagem fora do intervalo esperado foi a distribuição das classes. A conferência definitiva foi comparada com o CSV entregue e as contagens reproduzidas neste documento batem com o arquivo `nodulos_selecionados.csv`.
