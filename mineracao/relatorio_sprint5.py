"""Regenera a seção #31 apenas dos logs/predições salvos, sem carregar o NPZ."""
import csv
import hashlib
import json
import statistics
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RESULTADOS = RAIZ / 'mineracao/resultados'
SAIDA = RESULTADOS / 'sprint5'
INICIO = '<!-- inicio: issue31 -->'
FIM = '<!-- fim: issue31 -->'


def ler_rodada(pasta):
    resumo = json.loads((pasta / 'resumo.json').read_text())
    for nome, digest in resumo['arquivos_sha256'].items():
        if hashlib.sha256((pasta / nome).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Arquivo alterado: {pasta / nome}')
    linhas = (pasta / 'log_treino.txt').read_text().splitlines()
    registro = [json.loads(l.split('RESULTADO=', 1)[1]) for l in linhas if 'RESULTADO=' in l]
    if len(registro) != 1:
        raise ValueError(f'Log sem resultado único: {pasta}')
    r = registro[0]
    if any(resumo[k] != v for k, v in r.items()):
        raise ValueError(f'Resumo diverge do log: {pasta}')
    with (pasta / 'predicoes_validacao.csv').open() as f:
        pred = list(csv.DictReader(f))
    pos = [float(x['y_prob']) for x in pred if int(x['y_true']) == 1]
    neg = [float(x['y_prob']) for x in pred if int(x['y_true']) == 0]
    auc = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))
    if abs(auc - r['auc']) > 1e-12:
        raise ValueError(f'AUC diverge das predições: {pasta}')
    return r


def tabela(linhas, chave):
    t = f'| {chave} | AUC validação | Época | N treino | N validação | Pacientes validação |\n'
    t += '|---|---:|---:|---:|---:|---:|\n'
    for r in linhas:
        t += f"| {r[chave]} | {r['auc']:.6f} | {r['epoca']} | {r['n_treino']} | {r['n_validacao']} | {r['pacientes_validacao']} |\n"
    vals = [r['auc'] for r in linhas]
    t += f'| **Média** | **{statistics.mean(vals):.6f}** | — | — | — | — |\n'
    t += f'| **Desvio-padrão amostral** | **{statistics.stdev(vals):.6f}** | — | — | — | — |\n'
    t += f'\nMínimo–máximo: {min(vals):.6f}–{max(vals):.6f}.\n'
    return t


def main():
    p = json.loads((SAIDA / 'protocolo.json').read_text())
    sementes = [ler_rodada(RESULTADOS / 'real_com_pesos' if s == p['sementes'][0]
                 else SAIDA / 'sementes' / f'seed_{s}') for s in p['sementes']]
    cv = [ler_rodada(SAIDA / 'cv' / f'dobra_{d}') for d in range(1, p['n_dobras'] + 1)]
    sem = ler_rodada(RESULTADOS / 'real_sem_pesos')
    for r in sementes + cv + [sem]:
        if r['npz_sha256'] != p['npz_sha256']:
            raise ValueError('Hashes de dados divergentes entre rodadas')
    amb = p['ambiente']
    secao = f'''{INICIO}
## Sprint 5 — Variabilidade do baseline (Issue #31)

Esta seção é gerada por `python -m mineracao.relatorio_sprint5` a partir dos
logs e predições versionados. O comando confere a integridade dos arquivos e
recalcula a AUC de cada rodada; não carrega imagens nem dados de teste.

### Cinco sementes no split original

Sementes: {', '.join(map(str, p['sementes']))}. Mantidos CNN rasa, Adam,
lr={p['lr']}, batch={p['batch']}, {p['epocas']} épocas, pesos calculados no treino
e augmentation por flip 3D somente no treino. Muda apenas a semente.
O checkpoint escolhido minimiza a perda de validação, como no baseline.

Ambiente: Python {amb['python']}, PyTorch {amb['torch']}, dispositivo {amb['dispositivo']}.
SHA-256 do NPZ: `{p['npz_sha256']}`.
Versões e hashes completos: [protocolo](../mineracao/resultados/sprint5/protocolo.json).
As fontes exatas usadas nos treinos estão em `mineracao/resultados/sprint5/codigo_executado/`;
o patch de integração normaliza apenas o fim de arquivo das fontes originais.

{tabela(sementes, 'seed')}
### Validação cruzada agrupada por paciente

Treino e validação originais somam {sementes[0]['n_treino'] + sementes[0]['n_validacao']} nódulos.
Usou-se StratifiedGroupKFold com {p['n_dobras']} dobras e seed {p['seed_cv']}.
A guarda de separação por paciente de `mineracao/dataset.py` é reutilizada.
Cada paciente pertence a uma única dobra de avaliação; os pesos são calculados
somente no treino de cada dobra. As partições estão em
[dobras.csv](../mineracao/resultados/sprint5/dobras.csv).

A seed de treino permanece fixa. Na CV, avalia-se a época final {p['epocas']},
fixada antes das execuções: a dobra externa não seleciona checkpoint nem limiar.
Essa regra difere da seleção de época das cinco sementes; portanto, as médias
dos dois experimentos não isolam apenas o efeito da partição.

{tabela(cv, 'dobra')}
### Comparação com e sem pesos

A rodada ponderada usa a primeira semente acima e é reutilizada, sem novo treino.
A rodada sem pesos usa a mesma semente e a mesma seleção de checkpoint.

| Configuração | AUC validação | Época selecionada |
|---|---:|---:|
| Com pesos | {sementes[0]['auc']:.6f} | {sementes[0]['epoca']} |
| Sem pesos | {sem['auc']:.6f} | {sem['epoca']} |

A saída completa do script existente `comparar_pesos.py` está em
[comparacao_pesos.txt](../mineracao/resultados/sprint5/comparacao_pesos.txt).
Uma única semente não sustenta declarar uma configuração superior à outra.

### A AUC histórica de teste é típica ou foi sorte?

As cinco sementes mostram baixa dispersão no split de validação fixo; a CV
mostra maior variação entre grupos de pacientes. Os desvios acima são amostrais
(ddof=1), não intervalos de confiança. As dobras compartilham dados de treino.

A AUC histórica de teste apresentada na seção de comparação continua sendo
uma medida de uma execução. Estes experimentos não estimam sua variância:
as novas métricas usam validação e outros grupos de pacientes. Não é correto
somar o desvio da validação à AUC de teste, nem concluir que o resultado de
teste foi sorte por comparar médias de conjuntos diferentes. A conclusão
sustentada é estabilidade entre sementes nesta validação e sensibilidade à
partição por paciente; a tipicidade daquele resultado de teste permanece
indeterminada sem violar o congelamento estabelecido para esta issue.

Os testes de rótulos embaralhados da issue #30, já integrados na seção 9,
não apontaram sinal de vazamento nesse controle. Isso não prova ausência de
toda forma de vazamento nem remove as limitações de seleção do conjunto.
Nenhuma rodada desta issue avaliou o teste; os arquivos da comparação histórica
permanecem inalterados.
{FIM}
'''
    doc = RAIZ / 'docs/resultados_comparacao.md'
    texto = doc.read_text(encoding='utf-8')
    if INICIO in texto:
        antes, resto = texto.split(INICIO, 1)
        _, depois = resto.split(FIM, 1)
        texto = antes + secao.rstrip() + depois
    else:
        marcador = '## Referências'
        if marcador not in texto:
            raise ValueError('Marcador de referências ausente; revisar integração')
        texto = texto.replace(marcador, secao + '\n' + marcador, 1)
    doc.write_text(texto, encoding='utf-8')
    (SAIDA / 'secao_resultados.md').write_text(secao, encoding='utf-8')
    print('Seção gerada; AUCs das 11 rodadas conferidas nas predições.')


if __name__ == '__main__':
    main()
