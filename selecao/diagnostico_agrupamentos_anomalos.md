# Diagnostico dos agrupamentos anomalos — Issue #4

Gerado por `selecao/diagnostico_agrupamentos_anomalos.py`.

## O que o codigo do pylidc ja dizia

O `cluster_annotations()` tenta resolver este problema sozinho: quando algum agrupamento passa de 4 anotacoes, ele reduz a tolerancia em 10% e reagrupa, repetidamente, ate nenhum grupo exceder 4 — ou ate a tolerancia cair abaixo de 0,1 mm, quando imprime:

```
Failed to reduce all groups to <= 4 Annotations.
Some nodules may be close and must be grouped manually.
```

Os 9 agrupamentos anomalos sao os casos em que ele desistiu. O `selecao_nodulos.py` chama o metodo com `verbose=False`, o que silenciou esse aviso desde a primeira execucao.

**Consequencia que ainda nao estava documentada:** a tolerancia e reduzida para o exame inteiro, nao so para o agrupamento problematico. Nesses 9 exames ela caiu de ate 2,5 mm para 0,1 mm, e os demais nodulos do mesmo exame foram agrupados sob essa tolerancia minuscula. A coluna *agrupamentos soltos* abaixo mostra quantos ficaram com 1 ou 2 anotacoes e por isso foram descartados pelo filtro de consenso >= 3.

## Resumo

| nodulo | anot. | centroides max | contornos max | diametro | rotulo | separa em | pares repetidos | rotulo corrigido |
|---|---:|---:|---:|---:|---|---|---:|---|
| `LIDC-IDRI-0942_scan713_cluster000` | 7 | 22.25 mm | 1.414 mm | 27.54 mm | maligno | nao separa | 1 | maligno |
| `LIDC-IDRI-0204_scan205_cluster000` | 6 | 19.84 mm | 4.123 mm | 19.2 mm | maligno | nao separa | 1 | maligno |
| `LIDC-IDRI-0252_scan252_cluster000` | 5 | 18.83 mm | 1.000 mm | 27.49 mm | maligno | nao separa | 3 | maligno |
| `LIDC-IDRI-0608_scan613_cluster005` | 5 | 13.50 mm | 6.000 mm | 17.59 mm | maligno | nao separa | 0 | maligno |
| `LIDC-IDRI-0055_scan66_cluster000` | 6 | 13.24 mm | 3.606 mm | 13.43 mm | maligno | nao separa | 1 | EXCLUIDO (indeterminado) ⚠️ |
| `LIDC-IDRI-0137_scan140_cluster002` | 5 | 12.95 mm | 1.000 mm | 19.65 mm | maligno | nao separa | 3 | maligno |
| `LIDC-IDRI-0815_scan840_cluster000` | 5 | 12.83 mm | 4.000 mm | 13.38 mm | maligno | nao separa | 1 | maligno |
| `LIDC-IDRI-0865_scan790_cluster000` | 5 | 10.71 mm | 1.000 mm | 28.64 mm | maligno | nao separa | 5 | maligno |
| `LIDC-IDRI-0863_scan792_cluster001` | 5 | 4.28 mm | 2.000 mm | 8.34 mm | benigno | nao separa | 1 | benigno |

⚠️ = o rotulo muda quando as marcacoes repetidas sao removidas.

## As tres saidas, e quando cada uma se aplica

O §9 do `criterios_selecao.md` oferece duas saidas — descartar ou manter com limitacao. A tabela acima abre uma terceira.

1. **Descartar a linha.** Quando o grupo se separa em blocos com centroides distantes: sao lesoes diferentes, e a mediana esta misturando duas lesoes. Custo: o N cai e o split e a extracao precisam ser refeitos.
2. **Corrigir o agrupamento e manter.** Quando ha pares com centroide e diametro quase identicos e a mediana nao muda ao remove-los: era a mesma marcacao contada duas vezes. Preserva o N e corrige o `n_radiologistas`. **Exige inspecao visual dos contornos antes de aplicar** — centroide e diametro parecidos sao indicio, nao prova.
3. **Manter e declarar a limitacao.** Quando o grupo nao se separa e nao ha pares repetidos: uma lesao grande contornada em partes. A mediana continua descrevendo uma lesao so.

Decisao mista entre as tres e legitima, desde que cada linha tenha o motivo registrado. O que nao vale e decidir os nove em bloco sem olhar.

**Independente da escolha:** enquanto houver qualquer linha com mais de 4 anotacoes, toda descricao do criterio precisa dizer *"consenso de ao menos 3 radiologistas"*, nunca *"3 de 4"* — a segunda formulacao e falsa para esses casos. E o `selecao_nodulos.py` deveria passar a chamar `cluster_annotations(verbose=True)`, ou registrar em log os exames em que o pylidc desiste, para que a proxima execucao nao esconda o problema de novo.

## Caso a caso

### `LIDC-IDRI-0055_scan66_cluster000`

- `scan_id` 66 · agrupamento 0 · 6 anotacoes · diametro medio 13.43 mm
- escores: `2,3,4,4,4,2` → mediana 3.5 → **maligno**
- espacamento Z: 2.500 mm (slice_spacing (medido do DICOM)) · pixel_spacing: 0.7031 mm


**Pares de anotacoes**

| A | B | mal. A | mal. B | centroides (mm) | contornos (mm) |
|---|---|---:|---:|---:|---:|
| 584 | 585 | 4 | 4 | 13.24 | 3.606 |
| 572 | 584 | 2 | 4 | 12.64 | 3.606 |
| 577 | 584 | 3 | 4 | 12.63 | 2.236 |
| 579 | 585 | 4 | 4 | 12.57 | 3.317 |
| 572 | 579 | 2 | 4 | 11.89 | 3.000 |
| 577 | 579 | 3 | 4 | 11.81 | 3.000 |
| 584 | 588 | 4 | 2 | 8.46 | 0.000 |
| 579 | 588 | 4 | 2 | 7.22 | 0.000 |
| 585 | 588 | 4 | 2 | 6.30 | 0.000 |
| 572 | 588 | 2 | 2 | 5.46 | 0.000 |
| 577 | 588 | 3 | 2 | 5.30 | 0.000 |
| 579 | 584 | 4 | 4 | 2.39 | 0.000 |
| 577 | 585 | 3 | 4 | 1.16 | 0.000 |
| 572 | 585 | 2 | 4 | 0.88 | 0.000 |
| 572 | 577 | 2 | 3 | 0.38 | 0.000 |

Centroides: maxima **13.24 mm** · Contornos: minima 0.000 mm, maxima **3.606 mm**

**Varredura de limiares (sem o laco adaptativo do pylidc)**

| limiar | tamanho dos pedacos |
|---:|---|
| 0.5 mm | 6 |
| 0.2 mm | 6 |
| 0.1 mm | 6 |
| 0.05 mm | 6 |
| 0.02 mm | 6 |
| 0.01 mm | 6 |
| 0 mm | 6 |

**Candidatos a marcacao repetida**

| A | B | centroides (mm) | dif. de diametro |
|---|---|---:|---:|
| 572 | 585 | 0.88 | 9.1% |

Sem as repeticoes: 5 anotacoes, escores `2,3,4,4,2`, mediana 3 → **EXCLUIDO (indeterminado)** (era 3.5 → maligno)

**Contexto do exame:** 20 anotacoes em 7 agrupamentos. Tamanhos: {1: 2, 2: 2, 4: 2, 6: 1}. Agrupamentos com 1 ou 2 anotacoes: 4 (verificacao do efeito da tolerancia colapsada sobre os demais nodulos do exame).

**Leitura:** O grupo nao se separa em limiar algum, mas ha 1 par(es) com centroide e diametro quase identicos — candidatos a marcacao repetida. Removendo as repeticoes sobram 5 anotacoes, mediana 3 → **EXCLUIDO (indeterminado)**. **O rotulo MUDA**, entao esta linha nao pode ser mantida como esta — ou se corrige o agrupamento, ou se descarta.


### `LIDC-IDRI-0137_scan140_cluster002`

- `scan_id` 140 · agrupamento 2 · 5 anotacoes · diametro medio 19.65 mm
- escores: `3,2,4,4,5` → mediana 4 → **maligno**
- espacamento Z: 2.500 mm (slice_spacing (medido do DICOM)) · pixel_spacing: 0.7812 mm


**Pares de anotacoes**

| A | B | mal. A | mal. B | centroides (mm) | contornos (mm) |
|---|---|---:|---:|---:|---:|
| 1194 | 1195 | 4 | 5 | 12.95 | 1.000 |
| 1193 | 1194 | 4 | 4 | 8.70 | 0.000 |
| 1189 | 1194 | 2 | 4 | 8.29 | 0.000 |
| 1187 | 1194 | 3 | 4 | 7.97 | 0.000 |
| 1187 | 1195 | 3 | 5 | 5.09 | 0.000 |
| 1189 | 1195 | 2 | 5 | 4.75 | 0.000 |
| 1193 | 1195 | 4 | 5 | 4.39 | 0.000 |
| 1187 | 1193 | 3 | 4 | 0.83 | 0.000 |
| 1189 | 1193 | 2 | 4 | 0.46 | 0.000 |
| 1187 | 1189 | 3 | 2 | 0.40 | 0.000 |

Centroides: maxima **12.95 mm** · Contornos: minima 0.000 mm, maxima **1.000 mm**

**Varredura de limiares (sem o laco adaptativo do pylidc)**

| limiar | tamanho dos pedacos |
|---:|---|
| 0.5 mm | 5 |
| 0.2 mm | 5 |
| 0.1 mm | 5 |
| 0.05 mm | 5 |
| 0.02 mm | 5 |
| 0.01 mm | 5 |
| 0 mm | 5 |

**Candidatos a marcacao repetida**

| A | B | centroides (mm) | dif. de diametro |
|---|---|---:|---:|
| 1187 | 1189 | 0.40 | 5.4% |
| 1189 | 1193 | 0.46 | 9.9% |
| 1187 | 1193 | 0.83 | 4.8% |

Sem as repeticoes: 3 anotacoes, escores `3,4,5`, mediana 4 → **maligno** (era 4 → maligno)

**Contexto do exame:** 10 anotacoes em 3 agrupamentos. Tamanhos: {2: 1, 3: 1, 5: 1}. Agrupamentos com 1 ou 2 anotacoes: 1 (verificacao do efeito da tolerancia colapsada sobre os demais nodulos do exame).

**Leitura:** O grupo nao se separa em limiar algum, mas ha 3 par(es) com centroide e diametro quase identicos — candidatos a marcacao repetida. Removendo as repeticoes sobram 3 anotacoes, mediana 4 → **maligno**. O rotulo nao muda: **corrigir o agrupamento e manter** preserva o N sem alterar o dado.


### `LIDC-IDRI-0204_scan205_cluster000`

- `scan_id` 205 · agrupamento 0 · 6 anotacoes · diametro medio 19.2 mm
- escores: `4,3,5,5,4,4` → mediana 4.0 → **maligno**
- espacamento Z: 2.500 mm (slice_spacing (medido do DICOM)) · pixel_spacing: 0.7031 mm


**Pares de anotacoes**

| A | B | mal. A | mal. B | centroides (mm) | contornos (mm) |
|---|---|---:|---:|---:|---:|
| 1721 | 1722 | 4 | 4 | 19.84 | 4.123 |
| 1720 | 1722 | 5 | 4 | 14.07 | 1.000 |
| 1717 | 1721 | 4 | 4 | 11.91 | 0.000 |
| 1719 | 1721 | 5 | 4 | 11.74 | 0.000 |
| 1718 | 1721 | 3 | 4 | 10.83 | 0.000 |
| 1720 | 1721 | 5 | 4 | 9.46 | 0.000 |
| 1718 | 1722 | 3 | 4 | 9.33 | 0.000 |
| 1719 | 1722 | 5 | 4 | 9.31 | 0.000 |
| 1717 | 1722 | 4 | 4 | 8.12 | 0.000 |
| 1717 | 1720 | 4 | 5 | 6.72 | 0.000 |
| 1718 | 1720 | 3 | 5 | 5.59 | 0.000 |
| 1719 | 1720 | 5 | 5 | 5.02 | 0.000 |
| 1717 | 1719 | 4 | 5 | 2.20 | 0.000 |
| 1718 | 1719 | 3 | 5 | 1.87 | 0.000 |
| 1717 | 1718 | 4 | 3 | 1.29 | 0.000 |

Centroides: maxima **19.84 mm** · Contornos: minima 0.000 mm, maxima **4.123 mm**

**Varredura de limiares (sem o laco adaptativo do pylidc)**

| limiar | tamanho dos pedacos |
|---:|---|
| 0.5 mm | 6 |
| 0.2 mm | 6 |
| 0.1 mm | 6 |
| 0.05 mm | 6 |
| 0.02 mm | 6 |
| 0.01 mm | 6 |
| 0 mm | 6 |

**Candidatos a marcacao repetida**

| A | B | centroides (mm) | dif. de diametro |
|---|---|---:|---:|
| 1717 | 1718 | 1.29 | 3.3% |

Sem as repeticoes: 5 anotacoes, escores `4,5,5,4,4`, mediana 4 → **maligno** (era 4.0 → maligno)

**Contexto do exame:** 6 anotacoes em 1 agrupamentos. Tamanhos: {6: 1}. Agrupamentos com 1 ou 2 anotacoes: 0 (verificacao do efeito da tolerancia colapsada sobre os demais nodulos do exame).

**Leitura:** O grupo nao se separa em limiar algum, mas ha 1 par(es) com centroide e diametro quase identicos — candidatos a marcacao repetida. Removendo as repeticoes sobram 5 anotacoes, mediana 4 → **maligno**. O rotulo nao muda: **corrigir o agrupamento e manter** preserva o N sem alterar o dado.


### `LIDC-IDRI-0252_scan252_cluster000`

- `scan_id` 252 · agrupamento 0 · 5 anotacoes · diametro medio 27.49 mm
- escores: `5,3,5,4,4` → mediana 4 → **maligno**
- espacamento Z: 2.500 mm (slice_spacing (medido do DICOM)) · pixel_spacing: 0.8594 mm


**Pares de anotacoes**

| A | B | mal. A | mal. B | centroides (mm) | contornos (mm) |
|---|---|---:|---:|---:|---:|
| 1984 | 1985 | 4 | 4 | 18.83 | 1.000 |
| 1982 | 1985 | 3 | 4 | 13.30 | 0.000 |
| 1983 | 1985 | 5 | 4 | 11.67 | 0.000 |
| 1981 | 1985 | 5 | 4 | 11.44 | 0.000 |
| 1981 | 1984 | 5 | 4 | 7.40 | 0.000 |
| 1983 | 1984 | 5 | 4 | 7.19 | 0.000 |
| 1982 | 1984 | 3 | 4 | 5.57 | 0.000 |
| 1982 | 1983 | 3 | 5 | 1.96 | 0.000 |
| 1981 | 1982 | 5 | 3 | 1.90 | 0.000 |
| 1981 | 1983 | 5 | 5 | 0.75 | 0.000 |

Centroides: maxima **18.83 mm** · Contornos: minima 0.000 mm, maxima **1.000 mm**

**Varredura de limiares (sem o laco adaptativo do pylidc)**

| limiar | tamanho dos pedacos |
|---:|---|
| 0.5 mm | 5 |
| 0.2 mm | 5 |
| 0.1 mm | 5 |
| 0.05 mm | 5 |
| 0.02 mm | 5 |
| 0.01 mm | 5 |
| 0 mm | 5 |

**Candidatos a marcacao repetida**

| A | B | centroides (mm) | dif. de diametro |
|---|---|---:|---:|
| 1981 | 1983 | 0.75 | 9.6% |
| 1981 | 1982 | 1.90 | 8.5% |
| 1982 | 1983 | 1.96 | 1.2% |

Sem as repeticoes: 3 anotacoes, escores `5,4,4`, mediana 4 → **maligno** (era 4 → maligno)

**Contexto do exame:** 5 anotacoes em 1 agrupamentos. Tamanhos: {5: 1}. Agrupamentos com 1 ou 2 anotacoes: 0 (verificacao do efeito da tolerancia colapsada sobre os demais nodulos do exame).

**Leitura:** O grupo nao se separa em limiar algum, mas ha 3 par(es) com centroide e diametro quase identicos — candidatos a marcacao repetida. Removendo as repeticoes sobram 3 anotacoes, mediana 4 → **maligno**. O rotulo nao muda: **corrigir o agrupamento e manter** preserva o N sem alterar o dado.


### `LIDC-IDRI-0608_scan613_cluster005`

- `scan_id` 613 · agrupamento 5 · 5 anotacoes · diametro medio 17.59 mm
- escores: `5,5,2,4,4` → mediana 4 → **maligno**
- espacamento Z: 1.000 mm (slice_spacing (medido do DICOM)) · pixel_spacing: 0.6758 mm


**Pares de anotacoes**

| A | B | mal. A | mal. B | centroides (mm) | contornos (mm) |
|---|---|---:|---:|---:|---:|
| 4303 | 4312 | 5 | 2 | 13.50 | 3.000 |
| 4311 | 4312 | 5 | 2 | 12.74 | 5.000 |
| 4312 | 4316 | 2 | 4 | 12.60 | 6.000 |
| 4312 | 4313 | 2 | 4 | 10.50 | 0.000 |
| 4303 | 4313 | 5 | 4 | 3.50 | 0.000 |
| 4311 | 4313 | 5 | 4 | 2.49 | 0.000 |
| 4303 | 4316 | 5 | 4 | 2.47 | 0.000 |
| 4313 | 4316 | 4 | 4 | 2.12 | 0.000 |
| 4303 | 4311 | 5 | 5 | 1.49 | 0.000 |
| 4311 | 4316 | 5 | 4 | 1.40 | 0.000 |

Centroides: maxima **13.50 mm** · Contornos: minima 0.000 mm, maxima **6.000 mm**

**Varredura de limiares (sem o laco adaptativo do pylidc)**

| limiar | tamanho dos pedacos |
|---:|---|
| 0.5 mm | 5 |
| 0.2 mm | 5 |
| 0.1 mm | 5 |
| 0.05 mm | 5 |
| 0.02 mm | 5 |
| 0.01 mm | 5 |
| 0 mm | 5 |

**Candidatos a marcacao repetida**

Nenhum par com centroide a menos de 2 mm e diametro dentro de 10%. Nao ha indicio de marcacao repetida.

**Contexto do exame:** 16 anotacoes em 6 agrupamentos. Tamanhos: {1: 3, 4: 2, 5: 1}. Agrupamentos com 1 ou 2 anotacoes: 3 (verificacao do efeito da tolerancia colapsada sobre os demais nodulos do exame).

**Leitura:** O grupo **nao se separa em nenhum limiar testado** e nao ha pares candidatos a repeticao. Compativel com uma lesao grande contornada em partes por radiologistas diferentes. **Argumento para manter e declarar a limitacao** — a mediana continua descrevendo uma lesao so.


### `LIDC-IDRI-0942_scan713_cluster000`

- `scan_id` 713 · agrupamento 0 · 7 anotacoes · diametro medio 27.54 mm
- escores: `5,4,5,5,5,3,3` → mediana 5 → **maligno**
- espacamento Z: 1.000 mm (slice_spacing (medido do DICOM)) · pixel_spacing: 0.7422 mm


**Pares de anotacoes**

| A | B | mal. A | mal. B | centroides (mm) | contornos (mm) |
|---|---|---:|---:|---:|---:|
| 4884 | 4886 | 5 | 5 | 22.25 | 0.000 |
| 4886 | 4894 | 5 | 3 | 21.78 | 0.000 |
| 4884 | 4885 | 5 | 4 | 21.54 | 1.000 |
| 4884 | 4893 | 5 | 3 | 21.52 | 1.414 |
| 4886 | 4888 | 5 | 5 | 21.38 | 0.000 |
| 4885 | 4894 | 4 | 3 | 21.06 | 0.000 |
| 4893 | 4894 | 3 | 3 | 21.06 | 0.000 |
| 4888 | 4893 | 5 | 3 | 20.69 | 0.000 |
| 4885 | 4888 | 4 | 5 | 20.68 | 0.000 |
| 4884 | 4890 | 5 | 5 | 15.56 | 0.000 |
| 4890 | 4894 | 5 | 3 | 15.06 | 0.000 |
| 4888 | 4890 | 5 | 5 | 14.71 | 0.000 |
| 4886 | 4890 | 5 | 5 | 6.80 | 0.000 |
| 4885 | 4890 | 4 | 5 | 6.03 | 0.000 |
| 4890 | 4893 | 5 | 3 | 6.02 | 0.000 |
| 4886 | 4893 | 5 | 3 | 1.29 | 0.000 |
| 4884 | 4888 | 5 | 5 | 1.22 | 0.000 |
| 4884 | 4894 | 5 | 3 | 1.12 | 0.000 |
| 4885 | 4886 | 4 | 5 | 0.88 | 0.000 |
| 4888 | 4894 | 5 | 3 | 0.84 | 0.000 |
| 4885 | 4893 | 4 | 3 | 0.73 | 0.000 |

Centroides: maxima **22.25 mm** · Contornos: minima 0.000 mm, maxima **1.414 mm**

**Varredura de limiares (sem o laco adaptativo do pylidc)**

| limiar | tamanho dos pedacos |
|---:|---|
| 0.5 mm | 7 |
| 0.2 mm | 7 |
| 0.1 mm | 7 |
| 0.05 mm | 7 |
| 0.02 mm | 7 |
| 0.01 mm | 7 |
| 0 mm | 7 |

**Candidatos a marcacao repetida**

| A | B | centroides (mm) | dif. de diametro |
|---|---|---:|---:|
| 4884 | 4888 | 1.22 | 1.6% |

Sem as repeticoes: 6 anotacoes, escores `5,4,5,5,3,3`, mediana 4.5 → **maligno** (era 5 → maligno)

**Contexto do exame:** 14 anotacoes em 4 agrupamentos. Tamanhos: {1: 1, 2: 1, 4: 1, 7: 1}. Agrupamentos com 1 ou 2 anotacoes: 2 (verificacao do efeito da tolerancia colapsada sobre os demais nodulos do exame).

**Leitura:** O grupo nao se separa em limiar algum, mas ha 1 par(es) com centroide e diametro quase identicos — candidatos a marcacao repetida. Removendo as repeticoes sobram 6 anotacoes, mediana 4.5 → **maligno**. O rotulo nao muda: **corrigir o agrupamento e manter** preserva o N sem alterar o dado.


### `LIDC-IDRI-0865_scan790_cluster000`

- `scan_id` 790 · agrupamento 0 · 5 anotacoes · diametro medio 28.64 mm
- escores: `5,5,5,5,5` → mediana 5 → **maligno**
- espacamento Z: 2.500 mm (slice_spacing (medido do DICOM)) · pixel_spacing: 0.7031 mm


**Pares de anotacoes**

| A | B | mal. A | mal. B | centroides (mm) | contornos (mm) |
|---|---|---:|---:|---:|---:|
| 5358 | 5360 | 5 | 5 | 10.71 | 0.000 |
| 5357 | 5358 | 5 | 5 | 10.68 | 1.000 |
| 5356 | 5358 | 5 | 5 | 10.66 | 0.000 |
| 5358 | 5359 | 5 | 5 | 9.97 | 0.000 |
| 5357 | 5360 | 5 | 5 | 2.16 | 0.000 |
| 5356 | 5357 | 5 | 5 | 1.94 | 0.000 |
| 5357 | 5359 | 5 | 5 | 1.41 | 0.000 |
| 5359 | 5360 | 5 | 5 | 1.29 | 0.000 |
| 5356 | 5359 | 5 | 5 | 0.95 | 0.000 |
| 5356 | 5360 | 5 | 5 | 0.91 | 0.000 |

Centroides: maxima **10.71 mm** · Contornos: minima 0.000 mm, maxima **1.000 mm**

**Varredura de limiares (sem o laco adaptativo do pylidc)**

| limiar | tamanho dos pedacos |
|---:|---|
| 0.5 mm | 5 |
| 0.2 mm | 5 |
| 0.1 mm | 5 |
| 0.05 mm | 5 |
| 0.02 mm | 5 |
| 0.01 mm | 5 |
| 0 mm | 5 |

**Candidatos a marcacao repetida**

| A | B | centroides (mm) | dif. de diametro |
|---|---|---:|---:|
| 5356 | 5360 | 0.91 | 7.5% |
| 5356 | 5359 | 0.95 | 3.5% |
| 5359 | 5360 | 1.29 | 4.2% |
| 5357 | 5359 | 1.41 | 1.6% |
| 5356 | 5357 | 1.94 | 1.9% |

Sem as repeticoes: 2 anotacoes, escores `5,5`, mediana 5.0 → **maligno** (era 5 → maligno)

**Contexto do exame:** 5 anotacoes em 1 agrupamentos. Tamanhos: {5: 1}. Agrupamentos com 1 ou 2 anotacoes: 0 (verificacao do efeito da tolerancia colapsada sobre os demais nodulos do exame).

**Leitura:** O grupo nao se separa em limiar algum, mas ha 5 par(es) com centroide e diametro quase identicos — candidatos a marcacao repetida. Removendo as repeticoes sobram 2 anotacoes, mediana 5.0 → **maligno**. O rotulo nao muda: **corrigir o agrupamento e manter** preserva o N sem alterar o dado.


### `LIDC-IDRI-0863_scan792_cluster001`

- `scan_id` 792 · agrupamento 1 · 5 anotacoes · diametro medio 8.34 mm
- escores: `1,1,1,1,1` → mediana 1 → **benigno**
- espacamento Z: 0.700 mm (slice_spacing (medido do DICOM)) · pixel_spacing: 0.5859 mm


**Pares de anotacoes**

| A | B | mal. A | mal. B | centroides (mm) | contornos (mm) |
|---|---|---:|---:|---:|---:|
| 5367 | 5368 | 1 | 1 | 4.28 | 2.000 |
| 5368 | 5372 | 1 | 1 | 2.40 | 0.000 |
| 5365 | 5368 | 1 | 1 | 2.39 | 0.000 |
| 5368 | 5371 | 1 | 1 | 2.32 | 0.000 |
| 5365 | 5367 | 1 | 1 | 2.22 | 0.000 |
| 5367 | 5371 | 1 | 1 | 1.98 | 0.000 |
| 5367 | 5372 | 1 | 1 | 1.89 | 0.000 |
| 5365 | 5371 | 1 | 1 | 1.03 | 0.000 |
| 5365 | 5372 | 1 | 1 | 0.87 | 0.000 |
| 5371 | 5372 | 1 | 1 | 0.20 | 0.000 |

Centroides: maxima **4.28 mm** · Contornos: minima 0.000 mm, maxima **2.000 mm**

**Varredura de limiares (sem o laco adaptativo do pylidc)**

| limiar | tamanho dos pedacos |
|---:|---|
| 0.5 mm | 5 |
| 0.2 mm | 5 |
| 0.1 mm | 5 |
| 0.05 mm | 5 |
| 0.02 mm | 5 |
| 0.01 mm | 5 |
| 0 mm | 5 |

**Candidatos a marcacao repetida**

| A | B | centroides (mm) | dif. de diametro |
|---|---|---:|---:|
| 5367 | 5372 | 1.89 | 4.1% |

Sem as repeticoes: 4 anotacoes, escores `1,1,1,1`, mediana 1.0 → **benigno** (era 1 → benigno)

**Contexto do exame:** 8 anotacoes em 2 agrupamentos. Tamanhos: {3: 1, 5: 1}. Agrupamentos com 1 ou 2 anotacoes: 0 (verificacao do efeito da tolerancia colapsada sobre os demais nodulos do exame).

**Leitura:** O grupo nao se separa em limiar algum, mas ha 1 par(es) com centroide e diametro quase identicos — candidatos a marcacao repetida. Removendo as repeticoes sobram 4 anotacoes, mediana 1.0 → **benigno**. O rotulo nao muda: **corrigir o agrupamento e manter** preserva o N sem alterar o dado.


### `LIDC-IDRI-0815_scan840_cluster000`

- `scan_id` 840 · agrupamento 0 · 5 anotacoes · diametro medio 13.38 mm
- escores: `4,4,4,3,2` → mediana 4 → **maligno**
- espacamento Z: 1.000 mm (slice_spacing (medido do DICOM)) · pixel_spacing: 0.6641 mm


**Pares de anotacoes**

| A | B | mal. A | mal. B | centroides (mm) | contornos (mm) |
|---|---|---:|---:|---:|---:|
| 5711 | 5712 | 4 | 4 | 12.83 | 4.000 |
| 5712 | 5713 | 4 | 3 | 7.65 | 0.000 |
| 5710 | 5712 | 4 | 4 | 7.52 | 0.000 |
| 5712 | 5714 | 4 | 2 | 7.01 | 0.000 |
| 5711 | 5714 | 4 | 2 | 5.85 | 0.000 |
| 5711 | 5713 | 4 | 3 | 5.40 | 0.000 |
| 5710 | 5711 | 4 | 4 | 5.33 | 0.000 |
| 5710 | 5713 | 4 | 3 | 1.10 | 0.000 |
| 5713 | 5714 | 3 | 2 | 1.09 | 0.000 |
| 5710 | 5714 | 4 | 2 | 0.56 | 0.000 |

Centroides: maxima **12.83 mm** · Contornos: minima 0.000 mm, maxima **4.000 mm**

**Varredura de limiares (sem o laco adaptativo do pylidc)**

| limiar | tamanho dos pedacos |
|---:|---|
| 0.5 mm | 5 |
| 0.2 mm | 5 |
| 0.1 mm | 5 |
| 0.05 mm | 5 |
| 0.02 mm | 5 |
| 0.01 mm | 5 |
| 0 mm | 5 |

**Candidatos a marcacao repetida**

| A | B | centroides (mm) | dif. de diametro |
|---|---|---:|---:|
| 5713 | 5714 | 1.09 | 4.0% |

Sem as repeticoes: 4 anotacoes, escores `4,4,4,3`, mediana 4.0 → **maligno** (era 4 → maligno)

**Contexto do exame:** 5 anotacoes em 1 agrupamentos. Tamanhos: {5: 1}. Agrupamentos com 1 ou 2 anotacoes: 0 (verificacao do efeito da tolerancia colapsada sobre os demais nodulos do exame).

**Leitura:** O grupo nao se separa em limiar algum, mas ha 1 par(es) com centroide e diametro quase identicos — candidatos a marcacao repetida. Removendo as repeticoes sobram 4 anotacoes, mediana 4.0 → **maligno**. O rotulo nao muda: **corrigir o agrupamento e manter** preserva o N sem alterar o dado.
