# Auditoria do acervo local LIDC-IDRI

- **Responsável pela auditoria:** Wessel2007
- **Data da auditoria:** 12/09/2026
- **Acervo auditado:** `D:\lidc_idri` (baixado via NBIA Data Retriever, ~50 Mbps)
- **Configuração pylidc:** `%USERPROFILE%\pylidc.conf` com `path = D:\lidc_idri`

## Resumo

| Checagem | Resultado observado | Situação |
|---|---:|---|
| Pastas de paciente no acervo local | 1.010 | Conforme (bate com o manifesto do TCIA) |
| Pastas de paciente vazias (`n_dcm == 0`) | 71 | Requer atenção |
| Total de arquivos `.dcm` | 201.820 | Divergente do manifesto anterior (244.496) |
| Pastas de série duplicadas | 0 | Conforme |
| Nódulos com imagem disponível | 704 de 751 | Este é o N real da Sprint 3 |
| Leitura via `scan.to_volume()` (5 exames) | Sucesso em 5/5, ~2,43 s em média | Conforme |
| Faixa de HU observada | Ar/pulmão negativo, sem sinal de rescale duplicado | Conforme |

## 1. Contagem do acervo local

Percorrendo recursivamente `D:\lidc_idri` (script `selecao/gerar_acervo.py`), foram encontradas **1.010 pastas** no padrão `LIDC-IDRI-XXXX`, o que bate com o total de pacientes da coleção completa do TCIA.

- **Total de arquivos `.dcm`:** 201.820, contados em qualquer subpasta de cada paciente.
- **Pastas de paciente com `n_dcm == 0` (vazias):** 71. Ou seja, 71 pacientes têm a pasta criada mas nenhuma imagem baixada.
- **Pastas de série duplicadas** (mesma pasta-folha de série aparecendo duas vezes dentro do mesmo paciente): 0 ocorrências.

**Divergência encontrada:** o commit anterior (`selecao/acervo_disponivel.csv`, gerado por outro integrante a partir de uma cópia no Google Drive, `G:\Meu Drive\PI3-Grupo 01\Dataset\db\lidc_idri`) registrava **244.496 arquivos** para os mesmos 1.010 pacientes. A cópia local via NBIA Data Retriever auditada aqui tem **201.820 arquivos**, uma diferença de **42.676 arquivos (~17,5% a menos)**. Isso é consistente com as 71 pastas vazias encontradas e sugere que a cópia local está incompleta em relação à cópia usada para gerar o manifesto anterior — não há indício, porém, de arquivos corrompidos ou duplicados na cópia local.

## 2. Cruzamento com `nodulos_selecionados.csv`

Cruzando os 751 nódulos de `selecao/nodulos_selecionados.csv` (coluna `patient_id`) com os pacientes que têm `n_dcm > 0` em `acervo_disponivel.csv`:

- **Total de nódulos:** 751
- **Com imagem disponível (`tem_imagem = True`):** 704
- **Sem imagem disponível:** 47

Distribuição benigno/maligno **apenas entre os 704 nódulos disponíveis**:

| Rótulo | Quantidade |
|---|---:|
| Benigno | 354 |
| Maligno | 350 |

O resultado foi salvo em `selecao/nodulos_com_disponibilidade.csv`, com a coluna adicional `tem_imagem`.

## 3. Validação de leitura com pylidc

Foram escolhidos 5 `patient_id` distintos com `n_dcm > 0` (`LIDC-IDRI-0161`, `LIDC-IDRI-0867`, `LIDC-IDRI-0532`, `LIDC-IDRI-0683`, `LIDC-IDRI-0879`), com o pylidc configurado para `D:\lidc_idri`. Para cada um, `scan.to_volume()` foi executado e cronometrado:

| Patient ID | Tempo (s) | Shape do volume | HU mín. | HU máx. | HU mediana (slice central) |
|---|---:|---|---:|---:|---:|
| LIDC-IDRI-0161 | 2,05 | (512, 512, 251) | -2048 | 3071 | -909,0 |
| LIDC-IDRI-0867 | 4,03 | (512, 512, 477) | -3024 | 3071 | -749,0 |
| LIDC-IDRI-0532 | 3,04 | (512, 512, 330) | -1024 | 2134 | -406,0 |
| LIDC-IDRI-0683 | 1,09 | (512, 512, 138) | -3024 | 2007 | -815,0 |
| LIDC-IDRI-0879 | 1,95 | (512, 512, 261) | -2048 | 3071 | -881,0 |

**Tempo médio de leitura por exame:** 2,43 s.

**Faixa de HU:** os valores mínimos (-1024 a -3024) correspondem ao padrão de padding fora do campo de reconstrução do tomógrafo (comum em CT, valor abaixo do ar real de -1000 HU) e não a um erro de leitura. As medianas do slice central (-406 a -909 HU) são compatíveis com tecido pulmonar predominando no corte, com valores mais altos em cortes que atravessam mediastino/coração. Os máximos (até 3071 HU) são compatíveis com osso/calcificação e ficam dentro do intervalo típico de um CT com rescale aplicado uma única vez. **Não há sinal de rescale duplicado** (que produziria valores de ar fora da faixa de milhares, ou pulmão fora da faixa de centenas negativas).

## Conclusão

O acervo local tem a estrutura de pastas esperada (1.010 pacientes, padrão `LIDC-IDRI-XXXX`) e nenhuma pasta de série duplicada, mas **71 pastas de paciente estão vazias**, e o total de arquivos `.dcm` (201.820) é cerca de 17,5% menor que o registrado no manifesto anterior gerado a partir de outra cópia do acervo (244.496 arquivos). Isso indica que o download via NBIA Data Retriever pode estar incompleto para uma parte dos pacientes e deveria ser investigado/complementado antes de tratar o acervo local como definitivo.

Apesar disso, o cruzamento com a seleção de nódulos mostra que **704 dos 751 nódulos (93,7%) já têm imagem disponível localmente**, com distribuição benigno/maligno praticamente equilibrada (354/350). Esse é o N real de nódulos utilizáveis pela Sprint 3 com o acervo no estado atual.

A validação de leitura com `scan.to_volume()` funcionou em 5 de 5 exames testados, com tempos de leitura rápidos (~2,43 s em média) e faixas de HU fisiologicamente plausíveis, sem indício de rescale aplicado duas vezes. Portanto, a parte do acervo que está disponível é legível e consistente — o ponto a investigar é a cobertura incompleta (71 pastas vazias e o gap de ~42 mil arquivos frente ao manifesto anterior), não a integridade dos dados já baixados.
