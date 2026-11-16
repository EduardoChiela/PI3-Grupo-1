# PI3 — Grupo 01 · G3 Classificação

Projeto Integrador III: classificação binária de nódulos pulmonares previamente localizados, usando a coleção pública LIDC-IDRI. O repositório contém seleção, pré-processamento, transformação 2.5D, treinamento, avaliação e interpretação.

O alvo é a **avaliação de malignidade dos radiologistas**, não confirmação histopatológica. O sistema recebe o nódulo localizado; não detecta lesões nem possui uma classe “normal”.

## Estado do KDD

| Etapa | Estado e artefatos |
|---|---|
| Seleção | Critérios, exclusões e manifesto em [selecao/](selecao/) e [critérios de seleção](docs/criterios_selecao.md). |
| Pré-processamento | Divisão agrupada por paciente e pesos calculados somente no treino: [split_por_paciente.py](split_por_paciente.py) e [preprocessamento/](preprocessamento/). |
| Transformação | Patches 2.5D (axial, coronal, sagital), 3 × 64 × 64, normalizados para [0, 1]; [critérios](docs/criterios_transformacao.md) e [implementação](transformacao/extrai_patches.py). |
| Mineração | CNN rasa e ResNet18 treinadas; comparação, cinco sementes e CV do baseline em cinco dobras agrupadas por paciente. |
| Interpretação | Comparação pareada, IC 95% por paciente, piso do diâmetro, testes de estresse e limitações em [resultados_comparacao.md](docs/resultados_comparacao.md). |

As extensões da Sprint 6 são acompanhadas nas issues [#42](https://github.com/EduardoChiela/PI3-Grupo-1/issues/42), [#43](https://github.com/EduardoChiela/PI3-Grupo-1/issues/43) e [#45](https://github.com/EduardoChiela/PI3-Grupo-1/issues/45). A CV da ResNet e seus novos ICs estão na [PR #53](https://github.com/EduardoChiela/PI3-Grupo-1/pull/53); os respectivos scripts ficam disponíveis na `main` após a integração.

## Dados e resultados publicados

| Etapa | Unidade | Restantes | Exclusão |
|---|---|---:|---|
| Coleção completa | exames | 1.018 | — |
| Espessura ≤ 2,5 mm | exames | 897 | 121 exames |
| Agrupamentos de nódulos ≥ 3 mm | nódulos | 2.291 | — |
| Consenso ≥ 3 radiologistas | nódulos | 1.192 | 1.099 nódulos |
| Mediana de malignidade diferente de 3 | nódulos | 751 | 441 nódulos |
| Agrupamento estável | nódulos | 748 | 3 nódulos |
| Janela sem falha de espaçamento | nódulos | **747** | 1 nódulo |

Composição final: **387 benignos e 360 malignos**. Divisão: **522 treino, 114 validação e 111 teste**. O teste reúne **69 pacientes**.

`selecao/nodulos_selecionados.csv` preserva as 751 linhas originais. O conjunto de 747 resulta da aplicação das listas `excluidos_agrupamento_instavel.csv` e `excluidos_espacamento_irregular.csv`. O manifesto oficial consumido pelos modelos é `transformacao/manifest_patches.csv`.

**747 é o N metodológico.** A disponibilidade de DICOM em uma cópia local é outra medida: ver [auditoria do acervo](docs/auditoria_acervo.md). Não substitua esse N pela quantidade de imagens de uma cópia parcial.

Resultados de **teste holdout**, conforme [comparação da Sprint 5](docs/resultados_comparacao.md):

| Modelo / referência | AUC-ROC | Sensibilidade | Especificidade | Limiar |
|---|---:|---:|---:|---:|
| CNN rasa | 0,961 | 0,943 | 0,862 | 0,4305 |
| ResNet18 (layer3 + layer4) | 0,929 | 0,906 | 0,810 | 0,3614 |
| Diâmetro sozinho | 0,9463 | — | — | — |

Os limiares dos modelos foram escolhidos por Youden na validação e aplicados ao teste. As métricas e os ICs completos estão no documento; as AUCs acima não são médias de CV. A diferença CNN × ResNet não foi significativa no DeLong por paciente (p = 0,133). A referência de diâmetro evidencia a importância de controlar o tamanho do nódulo.

Limitações: duas classes de avaliação radiológica; exclusão dos casos ambíguos de mediana 3; ausência de validação externa e de explicabilidade; representação 2.5D e ResNet18; variabilidade numérica da ResNet em CPU. A comparação com o LUNA16 explica 4 dos 9 exames de diferença por espaçamento e mantém **5 sem explicação**. Consulte as ameaças em [critérios de seleção](docs/criterios_selecao.md) e as ressalvas em [critérios de mineração](docs/criterios_mineracao.md).

## Instalação e ambientes

O ambiente de referência é **CPython 3.12, CPU, Windows ou Linux x86-64**. Os pins estão em [requirements.txt](requirements.txt). Eles combinam versões registradas nos experimentos com escolhas explícitas de compatibilidade; **não representam um ambiente histórico único**. A origem de cada pin e os ambientes por etapa estão em [ambientes_reproducao.md](docs/ambientes_reproducao.md).

Em um checkout de trabalho, na raiz do projeto:

Windows / PowerShell, com Python 3.12 instalado:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Linux, com Python 3.12 instalado:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
source .venv/bin/activate
```

Nos comandos seguintes, `python` significa o executável desse ambiente. No PowerShell, pode-se usar `.\.venv\Scripts\python.exe` diretamente, sem ativar o ambiente.

A compatibilidade do pylidc 0.2.3 com Python 3.12+ foi tratada no commit [f2d4f1b](https://github.com/EduardoChiela/PI3-Grupo-1/commit/f2d4f1b0e4ae9c7035303f7ec22ae0573f6666b1): o `extrai_patches.py` restaura `SafeConfigParser` antes de importar pylidc. O diagnóstico e a orientação correspondente constam de [4e59ab9](https://github.com/EduardoChiela/PI3-Grupo-1/commit/4e59ab932de290ba80d3b767fdda8ae93241f4bb), em `selecao/verifica_espacamento.py`. Os shims de aliases antigos do NumPy também são preservados nos scripts existentes.

### Configuração do pylidc e acesso aos dados

Para consultar apenas anotações, crie `~/.pylidcrc` no Linux ou `%USERPROFILE%\pylidc.conf` no Windows:

```ini
[dicom]
warn = False
```

Para extrair imagens, acrescente `path` apontando para a raiz local dos pacientes LIDC-IDRI:

```ini
[dicom]
path = D:\lidc_idri
warn = True
```

No Linux, use o caminho correspondente do acervo. A estrutura dos subdiretórios deve seguir a [documentação do pylidc](https://pylidc.github.io/install.html). Os DICOMs não são versionados; a extração requer o acervo local. A consulta das anotações, o split e os relatórios baseados em predições salvas dispensam os 133 GB de imagens.

O `warn` do arquivo de configuração trata da localização dos DICOMs. É diferente do `verbose=False` de `cluster_annotations`, que permanece no código histórico e tem o diagnóstico explicado ao lado da chamada.

## Reprodução por etapa

**Use um checkout de trabalho.** Alguns comandos regeneram CSVs, figuras e documentos nos caminhos históricos. Preserve os artefatos congelados e confira `git diff` antes de versionar uma reexecução. As instruções de relatórios abaixo reaproveitam predições já salvas; reexecutar um treino é uma operação diferente.

### 1. Seleção e divisão por paciente

```bash
python -c "import pylidc as pl; print(pl.query(pl.Scan).count())"
python selecao/selecao_nodulos.py
python selecao/diagnostico_agrupamentos_anomalos.py
python selecao/verifica_espacamento.py
python split_por_paciente.py
python preprocessamento/pesos_classe.py
```

A consulta deve mostrar 1018 exames. A seleção grava o CSV bruto de 751 nódulos; o diagnóstico identifica os agrupamentos anômalos; a verificação de espaçamento gera seu CSV e relatório; o split grava `selecao/nodulos_com_split.csv` e `selecao/split_por_paciente_resumo.json`. O último comando confere pacientes e imprime os pesos. O filtro dos quatro nódulos excluídos é aplicado na transformação, sem sortear um novo split.

O `verbose=False` da seleção foi mantido para preservar o procedimento que gerou o CSV congelado. Os nove casos e as decisões estão em [diagnostico_agrupamentos_anomalos.md](selecao/diagnostico_agrupamentos_anomalos.md) e [critérios de seleção](docs/criterios_selecao.md).

### 2. Transformação

Teste sintético, sem DICOM:

```bash
python -c "from transformacao.extrai_patches import teste_sintetico; teste_sintetico()"
```

O ponto de entrada atual abaixo executa o teste sintético e extrai **somente o paciente LIDC-IDRI-0082**:

```bash
python transformacao/extrai_patches.py
```

Ele exige os DICOMs desse paciente e gera `patches/patches_2_5d_0082.npz` e `transformacao/manifest_patches_0082.csv`; registra falhas em `transformacao/falhas_patches_0082.csv` quando houver. Esse exemplo não gera o NPZ completo.

Para as etapas de modelos, use os artefatos oficiais versionados:

| Entrada | Conteúdo |
|---|---|
| `patches/patches_2_5d.npz` | 747 patches, forma (N, 3, 64, 64), float16, canais axial/coronal/sagital. |
| `transformacao/manifest_patches.csv` | Identidades, rótulos e split oficial de cada patch. |

SHA-256 do NPZ usado na comparação e na Sprint 5: `31a84904c9c39c97d362c46cbfef353afaecf3f531e68a90c1bfbf51d2a73fdd`. A geração completa não é o comportamento do `__main__` atual; os critérios e a rastreabilidade da extração estão em [criterios_transformacao.md](docs/criterios_transformacao.md). Não concatene apenas o exemplo do 0082 e o trate como o conjunto oficial.

Para refazer a medição da faixa preta usando o NPZ completo:

```bash
python transformacao/mede_faixa_preta.py
```

O comando grava `transformacao/patches_faixa_preta.csv` e confere o critério de padding.

### 3. Treinamento: novas execuções separadas das históricas

Os exemplos abaixo usam o split existente, 30 épocas e seed 42 e gravam em pastas novas. Eles não solicitam nova avaliação do teste:

```bash
python -m mineracao.treino_baseline --liberado-11 --npz patches/patches_2_5d.npz --manifesto transformacao/manifest_patches.csv --epocas 30 --seed 42 --saida mineracao/resultados/reexecucao/cnn
python -m mineracao.treino_resnet --npz patches/patches_2_5d.npz --manifesto transformacao/manifest_patches.csv --descongelar layer3_layer4 --lr-backbone 0.0001 --epocas 30 --seed 42 --saida mineracao/resultados/reexecucao/resnet
```

Produzem logs, métricas e predições de validação, ROC e pesos `.pt` (ignorados pelo Git). A seleção de checkpoint desses treinos usa a validação. Fixar dependências e seed não garante reproduzir os números históricos, em particular os da ResNet.

A avaliação do teste exige a opção explícita `--avaliar-teste`, usada no protocolo histórico após congelar as decisões. Não a acrescente a uma rodada exploratória. Para a comparação publicada, prefira os CSVs já salvos.

### 4. Avaliação e interpretação com os artefatos publicados

```bash
python -m mineracao.resumo_finetuning
python -m mineracao.comparacao --baseline mineracao/resultados/baseline --resnet mineracao/resultados/resnet_s5/final --manifesto transformacao/manifest_patches.csv --saida mineracao/resultados/comparacao_s5
python -m mineracao.relatorio_sprint5
```

| Comando | O que faz / saída |
|---|---|
| `resumo_finetuning` | Lê os logs das três configurações da ResNet e gera `resnet_s5/resumo_validacao.csv`, `resumo_validacao.md` e `escolha_validacao.json`. |
| `comparacao` | Confere CSVs de teste contra os logs, calcula ICs e testes pareados; grava tabela, JSON, CSV alinhado e ROC em `comparacao_s5/`. Não treina nem prevê. |
| `relatorio_sprint5` | Confere os hashes e as AUCs das sementes/dobras salvas e regenera tabelas, seção e documentação da Sprint 5, sem abrir o NPZ. |

A comparação deve apontar para `resnet_s5/final`. O padrão `resnet/` do script corresponde ao experimento antigo com backbone congelado.

### 5. Variabilidade, CV e testes de estresse

Conferência do desenvolvimento sem iniciar treinos:

```bash
python -m mineracao.experimentos_sprint5 --etapa verificar
```

As etapas `sementes` e `cv` desse módulo executam novos treinos e gravam nos diretórios da Sprint 5; suas guardas recusam mistura de configurações ou ambientes. Para reproduzir os números entregues, use `relatorio_sprint5` sobre os artefatos salvos. As fontes históricas também estão em `mineracao/resultados/sprint5/codigo_executado/`.

O teste de estresse é uma execução separada, com treinos de rótulos embaralhados e análise por tamanho. Para iniciá-lo conscientemente em um checkout de trabalho:

```bash
python -m mineracao.teste_estresse --npz patches/patches_2_5d.npz --manifesto transformacao/manifest_patches.csv --predicoes mineracao/resultados/comparacao_s5/predicoes_teste_pareadas.csv --saida mineracao/resultados/reexecucao/estresse
```

Grava `log_estresse.txt` e `resultados.json` na saída escolhida. Os resultados históricos e a distinção entre controle, embaralhamento e avaliação por tamanho estão nas seções 8–11 de [resultados_comparacao.md](docs/resultados_comparacao.md).

Após a integração da PR #53, a CV da ResNet pode ser conferida e seu relatório regenerado por:

```bash
python -m mineracao.experimentos_sprint6 --etapa verificar
python -m mineracao.relatorio_sprint6
```

A etapa `--etapa resnet` inicia cinco treinos de 30 épocas nas dobras congeladas; a verificação e o relatório são operações distintas dessa etapa.

## Documentação e responsabilidades

| Documento | Assunto |
|---|---|
| [Critérios de seleção](docs/criterios_selecao.md) | Filtros, rótulos, exclusões, comparação com o LUNA16 e ameaças. |
| [Critérios de transformação](docs/criterios_transformacao.md) | ROI, reamostragem, canais 2.5D, janela HU e faixa preta. |
| [Critérios de mineração](docs/criterios_mineracao.md) | Arquiteturas, augmentation, treinamento e decisões. |
| [Protocolo de avaliação](docs/protocolo_avaliacao.md) | Métricas, thresholds, ICs e validação das predições. |
| [Ambientes de reprodução](docs/ambientes_reproducao.md) | Pins, evidências históricas e limites de reexecução. |

As responsabilidades por entrega ficam nas issues atribuídas de cada sprint. O `split_por_paciente.py` permanece na raiz para preservar os caminhos existentes.

## Fonte e licença dos dados

LIDC-IDRI, *The Cancer Imaging Archive*, CC BY 3.0: Armato SG et al. (2011), *The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): a completed reference database of lung nodules on CT scans*, Medical Physics, 38(2), 915–931.

As anotações e manifestos são públicos e anonimizados. Os DICOMs não são incluídos neste repositório; `.gitignore` cobre imagens médicas, dados locais e pesos de modelos.
