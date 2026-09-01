# PI3 — Grupo 01 · G3 Classificação

Projeto Integrador III — **classificação de nódulos pulmonares** a partir da coleção pública LIDC-IDRI.

Este repositório guarda o código, os manifestos e a documentação do subgrupo **G3 · Classificação**. As etapas concluídas até aqui são as duas primeiras do pipeline KDD do grupo: **Seleção** e **Pré-processamento (Balanceamento de classes)**.

---

## Resultado atual

| Etapa | Unidade | Restantes | Excluídos | Motivo |
|---|---|---|---|---|
| Acervo completo | exames | 1.018 | — | — |
| Exames elegíveis | exames | 897 | 121 | espessura de corte > 2,5 mm |
| Agrupamentos de nódulo ≥ 3 mm | nódulos | 2.291 | n/d | lesões sem escore de malignidade não são contabilizadas |
| Consenso ≥ 3 radiologistas | nódulos | 1.192 | 1.099 | marcados por menos de 3 radiologistas |
| **Nódulos rotulados** | nódulos | **751** | 441 | malignidade mediana = 3 |

**Composição:** 388 benignos (51,66%) · 363 malignos (48,34%).

**Conferência externa:** o desafio LUNA16 reporta 888 exames e 1.186 nódulos sob filtragem equivalente — diferenças de 1,0% e 0,5% em relação aos nossos números.

---

## Estrutura

```
PI3-Grupo-1/
├── README.md
├── .gitignore
├── LICENSE
├── split_por_paciente.py           # divisão treino/validação/teste sem vazamento
├── selecao/
│   ├── selecao_nodulos.py          # aplica os critérios e gera o manifesto
│   └── nodulos_selecionados.csv    # 751 linhas, uma por nódulo elegível
├── preprocessamento/
│   ├── pesos_classe.py             # verifica o split e calcula os pesos de classe
│   └── estrategia_balanceamento.md # estratégia adotada e justificativa
└── docs/
    ├── criterios_selecao.md        # critérios, justificativas, limitações e Métodos
    └── conferencia_numeros.md      # auditoria dos números e achados de qualidade
```

> **Pendência de organização:** o `split_por_paciente.py` está na raiz, enquanto os outros dois arquivos da etapa de balanceamento estão em `preprocessamento/`. O natural é movê-lo para junto deles, mas a mudança precisa ser combinada com quem escreveu o arquivo, por causa de eventuais caminhos relativos no código.

---

## Como reproduzir

**Não é necessário baixar os 133 GB de DICOM para reproduzir a etapa de Seleção.** A biblioteca `pylidc` embute o banco de anotações e permite consultá-lo sem acesso às imagens.

```bash
pip install pylidc pandas scikit-learn
```

Crie o arquivo `~/.pylidcrc` (Linux/macOS) ou `pylidc.conf` em `%USERPROFILE%` (Windows), **sem a linha `path`**:

```ini
[dicom]
warn = False
```

Confirme que o banco está completo antes de qualquer filtro:

```python
import pylidc as pl
print(pl.query(pl.Scan).count())   # deve retornar 1018
```

Execute na ordem:

```bash
python selecao/selecao_nodulos.py       # gera nodulos_selecionados.csv
python split_por_paciente.py            # divide treino/validação/teste por paciente
python preprocessamento/pesos_classe.py # verifica vazamento e calcula os pesos
```

O acervo de imagem só passa a ser necessário na etapa seguinte, **Transformação**, quando for preciso ler o pixel para reamostrar e recortar os patches.

---

## Ordem que não pode ser invertida

**Dividir primeiro, balancear depois, e apenas o conjunto de treino.**

Se o balanceamento for aplicado antes da divisão, cópias da mesma amostra caem em conjuntos diferentes e a métrica final passa a medir memorização em vez de aprendizado. Não há correção posterior — apenas refazer. Validação e teste permanecem na proporção real, sob pena de as métricas deixarem de ter significado clínico.

---

## Limitação central

A variável-alvo do LIDC-IDRI (`malignancy`) é a **probabilidade de malignidade estimada por radiologistas a partir da imagem**, com variabilidade interobservador registrada — **não** confirmação histopatológica.

Um modelo treinado sobre este conjunto aprende a reproduzir a avaliação radiológica, não a detectar câncer confirmado. Toda conclusão deve ser enunciada nesses termos. As demais ameaças à validade estão documentadas em [`docs/criterios_selecao.md`](docs/criterios_selecao.md), seção 10.

---

## Divisão de responsabilidades

| Etapa | Faz | Confere | Documenta |
|---|---|---|---|
| Seleção | Luiz Wessel | Rafael Casarin | Áquila Begozzi |
| Balanceamento | Eduardo Chiela | Leticia Victoria | William Ames |

---

## Fonte dos dados

Coleção **LIDC-IDRI**, *The Cancer Imaging Archive*, licença **CC BY 3.0**.
1.010 pacientes · 1.018 exames · 244.527 imagens · anotações independentes de 4 radiologistas torácicos por exame.

> Armato SG et al. (2011). *The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): a completed reference database of lung nodules on CT scans.* Medical Physics, 38(2), 915–931.

**O acervo de imagem não é versionado neste repositório.** São 133 GB de DICOM, acima de qualquer limite prático do GitHub. O `.gitignore` deve conter `*.dcm`, `*.mhd`, `*.raw` e `data/` antes do primeiro commit de código. O manifesto CSV pode ser publicado sem restrição: a coleção é pública, anonimizada e os identificadores são do tipo `LIDC-IDRI-0001`.
