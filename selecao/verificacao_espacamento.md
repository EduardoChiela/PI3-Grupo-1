# Verificacao do criterio §3.2 — espacamento entre cortes

Gerado por `selecao/verifica_espacamento.py`, a partir das posicoes Z que o
pylidc guarda no proprio banco de anotacoes (`scan.slice_zvals`). **Nao usa
arquivo DICOM nenhum.**

## Resposta a pergunta da issue

O filtro de espacamento **nao estava aplicado nem verificado**: `selecao_nodulos.py` filtra somente `slice_thickness <= 2.5` e nao le posicao de corte em ponto algum. Esta e a primeira medicao.

## Numeros

| Situacao | Exames | % |
|---|---:|---:|
| Regular — passo constante e igual a espessura | 601 | 67.0% |
| Sobreposto — passo < espessura · **nao e defeito** | 292 | 32.6% |
| Espacado — passo > espessura, mas constante | 0 | 0.0% |
| **Irregular** — passo variavel ou corte ausente · **defeito** | 4 | 0.4% |
| Sem posicoes de corte registradas | 0 | 0.0% |
| **Total** | **897** | |

## Se o §3.2 virar criterio de exclusao

- Exames elegiveis hoje (so espessura): **897**
- Excluidos por espacamento irregular: **4**
- Exames restantes: **893**
- LUNA16 reporta: **888**
- Nodulos do manifesto nesses exames: **2**

> 893 ainda fica 5 exames acima dos 888 do LUNA16. O filtro de espacamento explica **parte** da diferenca, nao toda. A secao 8 precisa dizer isso: atribuir a diferenca inteira a este filtro seria inventar uma causa.

## O custo da leitura ingenua

A redacao antiga do §3.2 mandava sinalizar "divergencia acentuada entre `slice_spacing` e `slice_thickness`". Aplicada ao pe da letra, ela excluiria tambem os exames de reconstrucao sobreposta:

- Exames excluidos: **295** de 897
- Nodulos perdidos: **234** de 751

Sobreposicao nao e defeito — e o protocolo entregando mais informacao em Z, nao menos. Sao 292 exames que nao ha razao para descartar. Esta e a razao de o §3.2 ter sido reescrito separando os dois fenomenos.

## Exames irregulares

| Paciente | Espessura | Passo mediano | Maior desvio | Passos fora | Cortes |
|---|---:|---:|---:|---:|---:|
| `LIDC-IDRI-0514` | 2.5 mm | 2.0 mm | **36.0 mm** | 1 | 142 |
| `LIDC-IDRI-0267` | 2.5 mm | 2.5 mm | **2.0 mm** | 2 | 150 |
| `LIDC-IDRI-0123` | 1.25 mm | 1.25 mm | **0.75 mm** | 3 | 236 |
| `LIDC-IDRI-0672` | 1.25 mm | 0.625 mm | **0.625 mm** | 3 | 462 |

Tolerancia usada: 0.1 mm de desvio maximo em relacao ao passo mediano.

Detalhe por exame: `selecao/espacamento_por_exame.csv`.
