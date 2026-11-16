# Ambientes de reprodução — Issue #46

## Escopo e referência

Levantamento da `main` no commit [0751c08](https://github.com/EduardoChiela/PI3-Grupo-1/commit/0751c08dd26526cd78593f267669091b9e0dc411). O projeto teve ambientes diferentes entre etapas; um único `pip freeze` não descreve todos os números publicados.

O `requirements.txt` define um **ambiente de referência CPU com CPython 3.12**, partindo das versões registradas na Sprint 5 e completando as dependências de seleção, transformação e figuras. Os pins escolhidos para completar esse ambiente são identificados abaixo. Isso não transforma as execuções anteriores em execuções feitas nesse ambiente.

## Ambientes históricos com evidência

“Não registrado” significa que o artefato citado não permite afirmar a versão. Não foi preenchido com a versão do ambiente atual.

| Execução | Python / SO | Versões efetivamente registradas | Evidência |
|---|---|---|---|
| CNN da comparação histórica | 3.13.2 / Windows 10 | NumPy 2.4.4; pandas 3.0.3; torch 2.11.0+cpu | [Primeira linha do log](../mineracao/resultados/baseline/log_treino.txt) e [comparação, §1](resultados_comparacao.md). |
| ResNet inicial e fine-tuning da Sprint 5 | 3.12.10 / SO não registrado no cabeçalho | torch 2.14.1+cpu; torchvision 0.29.1+cpu; NumPy e pandas não registrados | [Log inicial](../mineracao/resultados/resnet/log_treino.txt) e [log final](../mineracao/resultados/resnet_s5/final/log_treino.txt). |
| CNN, cinco sementes e CV da Sprint 5 | 3.12.14 / Linux x86-64 | NumPy 2.3.5; pandas 2.2.3; scikit-learn 1.8.0; torch 2.14.1+cpu; 4 threads | [Protocolo completo](../mineracao/resultados/sprint5/protocolo.json), também incorporado aos logs. |
| Testes de estresse históricos | Python e SO não informados no trecho citado | torch 2.7.1 | [Resultados, §10.1](resultados_comparacao.md), na discussão da diferença do controle. |
| CV da ResNet da Sprint 6, PR #53 | 3.13.14 / Windows 11 | NumPy 2.5.4; pandas 3.1.1; torch 2.14.1+cpu; 4 threads | [Protocolo no commit da entrega](https://github.com/EduardoChiela/PI3-Grupo-1/blob/1c91b021536e2be97841c4c554bfb7bf964c7844/mineracao/resultados/sprint6/protocolo.json). É uma execução da PR #53, não um artefato já integrado à main no levantamento. |
| Seleção / banco de anotações | Python e demais versões não registrados integralmente | pylidc 0.2.3 | [Conferência dos números](conferencia_numeros.md) e shims nos scripts de seleção. |

O docstring de `mineracao/treino_baseline.py` cita outro ambiente (Python 3.14.5). Ele é uma declaração sobre o desenvolvimento do script, não prova de que todas as execuções posteriores o usaram. Para atribuir uma versão a um resultado, priorize o log e o protocolo daquela execução. Essa divergência foi preservada como histórico.

## Pins do ambiente de referência

| Biblioteca | Pin | Origem / motivo |
|---|---|---|
| NumPy | 2.3.5 | Registrado no protocolo da Sprint 5. |
| pandas | 2.2.3 | Registrado no protocolo da Sprint 5. |
| scikit-learn | 1.8.0 | Registrado como `sklearn` no protocolo da Sprint 5. |
| torch | 2.14.1+cpu | Registrado na Sprint 5 e nos logs da ResNet. CPU é explícito. |
| torchvision | 0.29.1+cpu | Registrado nos logs da ResNet, usado com o torch acima. |
| pylidc | 0.2.3 | Versão documentada na seleção. |
| pydicom | 2.4.4 | **Escolha de referência**, não versão histórica comprovada. Mantém a API `read_file` usada pelo pylidc 0.2.3. |
| SQLAlchemy | 1.4.54 | **Escolha de referência**, mantendo a API ORM legada do pylidc. |
| setuptools | 78.1.0 | **Escolha de referência**, fornecendo `pkg_resources`, importado pelo pylidc 0.2.3. A instalação inicialmente resolveu uma versão recente sem esse módulo; o import detectou a incompatibilidade. |
| scipy | 1.17.0 | **Escolha de referência** para interpolação, distâncias e dependências científicas. |
| scikit-image | 0.25.2 | **Escolha de referência** para as funções de imagem usadas pelo pylidc. |
| matplotlib | 3.10.8 | **Escolha de referência** para as figuras. A versão dos gráficos históricos não está integralmente registrada. |
| SimpleITK | 2.5.3 | **Escolha de referência**, incluída no levantamento solicitado pela #46. O `extrai_patches.py` atual interpola com SciPy e não importa SimpleITK; não há evidência suficiente para atribuir esse pin à extração histórica. |

Os pins são de dependências diretas e das dependências científicas/ORM relevantes. O arquivo não é um lock completo de todas as transitivas nem fixa drivers ou bibliotecas do sistema. O índice adicional oficial do PyTorch resolve os pacotes com sufixo `+cpu`.

## Compatibilidade documentada

- **Python 3.12+ / pylidc:** `configparser.SafeConfigParser` foi removido. [f2d4f1b](https://github.com/EduardoChiela/PI3-Grupo-1/commit/f2d4f1b0e4ae9c7035303f7ec22ae0573f6666b1) adicionou o alias para `ConfigParser` antes da importação de pylidc em `transformacao/extrai_patches.py`. [4e59ab9](https://github.com/EduardoChiela/PI3-Grupo-1/commit/4e59ab932de290ba80d3b767fdda8ae93241f4bb) registrou o diagnóstico e a orientação em `selecao/verifica_espacamento.py`.
- **NumPy recente / pylidc:** os aliases `np.int`, `np.float` e `np.bool` são tratados pelos shims existentes. Importe o módulo do projeto que contém o shim antes de chamar as rotinas correspondentes de pylidc.
- **pydicom:** o pin 2.4.4 conserva `read_file`. A escolha evita depender de uma API removida na série 3.x.
- **setuptools:** o pin 78.1.0 mantém `pkg_resources`, necessário para localizar os recursos do pylidc 0.2.3. `pip check` sozinho não detecta essa dependência implícita; a importação foi incluída na validação.
- **Agrupamento de anotações:** `verbose=False` foi mantido. O comentário junto à chamada aponta para a investigação dos nove agrupamentos; adicionar uma explicação não altera o algoritmo que produziu os CSVs.

Fontes externas: [instalação e configuração do pylidc](https://pylidc.github.io/install.html), [código de Scan](https://pylidc.github.io/_modules/pylidc/Scan.html), [pydicom 2.4.4](https://pypi.org/project/pydicom/2.4.4/), [SQLAlchemy 1.4.54](https://pypi.org/project/SQLAlchemy/1.4.54/) e [SimpleITK 2.5.3](https://pypi.org/project/simpleitk/2.5.3/).

## O que reproduzir

1. **Regeneração de resultados publicados:** ler os CSVs e logs congelados e refazer métricas, ICs, tabelas e figuras. É a rota principal de conferência da entrega.
2. **Nova execução de treino:** executar o código em uma pasta nova e registrar o ambiente dessa execução. O resultado é uma nova execução; não substitui silenciosamente um artefato histórico.
3. **Extração a partir de DICOM:** exige as imagens e sua configuração local. O ponto de entrada atual da transformação é um exemplo de um paciente; não produz sozinho o acervo completo.

A ResNet apresentou variação de 0,0154 de AUC de validação entre duas execuções em CPU com a mesma configuração e seed, conforme [critérios de mineração](criterios_mineracao.md). Por isso, nem o pin de versões nem a seed sustentam prometer identidade numérica entre novos treinos.

As guardas de hashes e configurações das Sprints 5 e 6 devem ser respeitadas. Uma alteração de ambiente ou de código pode impedir reutilização das saídas antigas. O `requirements.txt` não autoriza apagar protocolos nem reavaliar o teste para obter números “melhores”.

## Validação desta entrega

Verificações executadas em ambiente virtual isolado, **Linux x86-64 e CPython 3.12.14**, com as 13 versões finais de `requirements.txt` conferidas por `importlib.metadata`:

| Verificação | Resultado |
|---|---|
| Instalação das dependências e `python -m pip check` | Instalação concluída; nenhuma dependência quebrada após fixar setuptools 78.1.0. |
| Importação de pylidc, pydicom, SimpleITK, torch e torchvision | Concluída; `pydicom.read_file` e o shim de `SafeConfigParser` presentes. |
| Consulta independente `pl.query(pl.Scan).count()` | 1.018 exames no banco de anotações. |
| `transformacao.extrai_patches.teste_sintetico()` | Patch (3, 64, 64), float16 e intensidades dentro de [0, 1]. |
| CNN rasa sobre dois tensores aleatórios | Saída (2, 1), sem imagens clínicas ou treinamento. |
| `--help` de nove CLIs | Split, pesos, treino CNN, treino ResNet, comparação, experimentos S5, estresse, gerador sintético e comparação de pesos concluíram sem erro. |
| Split e pesos sobre 160 nódulos sintéticos de 80 pacientes | 112/24/24 nódulos; nenhum paciente em mais de um split; cálculo de pesos concluído. |
| `python -m unittest discover -s tests -v` | Os 6 testes existentes da Sprint 5 passaram, em dados sintéticos. |
| `python -m mineracao.resumo_finetuning` | Resumo das três configurações regenerado a partir dos logs. |
| `python -m mineracao.comparacao ... --resnet mineracao/resultados/resnet_s5/final` | Métricas, ICs e diferenças bootstrap reproduzidos exatamente dos CSVs; testes pareados equivalentes com tolerância de 1e-12. O DeLong diferiu apenas na precisão de ponto flutuante (ordem de 1e-16). |
| `python -m mineracao.relatorio_sprint5` | Concluiu e conferiu as AUCs das 11 rodadas nas predições. |
| AST da seleção antes/depois | Idêntica; a alteração no `.py` contém somente comentários, com `verbose=False` preservado. |

Esta validação não refez a seleção histórica, a extração de DICOM nem os treinos oficiais. O espelho de auditoria não incluiu o NPZ binário; portanto, a comparação foi refeita das predições salvas sem conferir localmente os bytes desse NPZ. A configuração Windows e o download dos pesos ImageNet não foram exercitados nesta validação Linux. A reexecução dos relatórios ocorreu na cópia de trabalho e seus arquivos gerados não fazem parte da alteração da #46.
