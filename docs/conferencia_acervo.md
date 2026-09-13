# Conferência das cópias do acervo no Drive

**Responsável:** Rafael Casarin  
**Data da conferência:** 13/09/2026

## 1. Objetivo

Conferir o estado atual das cópias do acervo LIDC-IDRI utilizadas pelo grupo, verificando a quantidade de pacientes e arquivos DICOM, a existência de pastas vazias, a possível duplicação de séries e quais cuidados devem ser tomados antes de qualquer limpeza no Google Drive.

---

## 2. Cópia conferida

A cópia atualmente utilizada e acessível pelo grupo foi localizada em:

`PI3-Grupo 01/Dataset/db/lidc_idri`

A contagem foi realizada de forma independente no Google Colab, percorrendo as pastas `LIDC-IDRI-*` e contando recursivamente os arquivos `.dcm`.

### Resultado

| Verificação | Resultado |
|---|---:|
| Pastas de pacientes | 1.010 |
| Arquivos `.dcm` | 244.496 |
| Pastas de pacientes vazias | 0 |

Os números obtidos correspondem à referência de acervo íntegro utilizada pelo grupo.

---

## 3. Caminho citado anteriormente

A issue mencionava a cópia:

`PI3-Grupo 01/Dataset/lidc_idri`

No momento desta conferência, esse caminho não estava presente diretamente dentro de `Dataset`.

A cópia encontrada e validada está atualmente em:

`PI3-Grupo 01/Dataset/db/lidc_idri`

Por isso, esta conferência registra o estado atual observado no Drive, sem assumir que o caminho anterior ainda representa a estrutura atual das pastas.

---

## 4. Verificação da `LIDC-IDRI-0082`

A pasta `LIDC-IDRI-0082` foi conferida diretamente pela interface web do Google Drive.

Foram observadas as seguintes pastas de série:

- `79176`
- `96503`

Na cópia verificada, apareceu **uma ocorrência de cada pasta**.

Portanto, a duplicidade descrita anteriormente na issue **não foi reproduzida no estado atual da cópia conferida**.

Isso não prova que a duplicidade nunca existiu; apenas registra que, durante esta conferência, ela não estava presente da forma descrita.

---

## 5. Arquivo de controle do download

O arquivo `completion_status.csv`, utilizado como registro do processo de download das séries, está preservado no repositório em:

`selecao/completion_status.csv`

Assim, a evidência de quais séries foram registradas como concluídas permanece versionada no projeto.

---

## 6. Decisão sobre limpeza das cópias

Com base na conferência realizada:

| Cópia / item | Situação | Ação recomendada |
|---|---|---|
| `PI3-Grupo 01/Dataset/db/lidc_idri` | Íntegra: 1.010 pacientes, 244.496 DICOMs e 0 pastas vazias | **MANTER** |
| `PI3-Grupo 01/Dataset/lidc_idri` | Não encontrada nesse caminho durante a conferência | Verificar se foi movida ou reorganizada |
| Cópias antigas/truncadas | Não foram novamente validadas nesta conferência | **NÃO APAGAR sem confirmação do grupo** |
| `completion_status.csv` | Preservado no repositório | **MANTER** |

Nenhuma exclusão foi realizada durante esta conferência.

---

## 7. Conclusão

A cópia atualmente validada em `PI3-Grupo 01/Dataset/db/lidc_idri` está completa de acordo com a referência utilizada pelo grupo, contendo **1.010 pacientes, 244.496 arquivos `.dcm` e nenhuma pasta de paciente vazia**.

A verificação manual da `LIDC-IDRI-0082` mostrou apenas uma pasta `79176` e uma pasta `96503`, portanto a duplicidade descrita anteriormente não foi reproduzida no estado atual da cópia inspecionada.

O arquivo `selecao/completion_status.csv` está preservado no repositório. Por segurança, nenhuma cópia antiga deve ser removida até que o grupo confirme explicitamente qual delas pode ser descartada.
