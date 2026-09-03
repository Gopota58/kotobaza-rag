#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Второй проход: в строках, где фигурирует кличка 'Кудрявая жопа',
согласуем зависимые слова в женском роде (местоимения, краткие прилагательные,
глаголы прошедшего времени после клички, полные прилагательные перед ней)."""
import re, os, glob
import inspect
if not hasattr(inspect, 'getargspec'):
    def _getargspec(func):
        f = inspect.getfullargspec(func)
        return (f.args, f.varargs, f.varkw, f.defaults)
    inspect.getargspec = _getargspec

import warnings
warnings.filterwarnings('ignore')
import pymorphy2

morph = pymorphy2.MorphAnalyzer()

DOCS = r'C:\rag_project\docs'

PRON = {'он': 'она', 'его': 'её', 'ему': 'ей', 'им': 'ей', 'нём': 'ней', 'ним': 'ней'}
POSS = {
    'мой': 'моя', 'моего': 'моей', 'моему': 'моей', 'моим': 'моей', 'моём': 'моей',
    'твой': 'твоя', 'твоего': 'твоей', 'твоему': 'твоей', 'твоим': 'твоей', 'твоём': 'твоей',
    'свой': 'своя', 'своего': 'своей', 'своему': 'своей', 'своим': 'своей', 'своём': 'своей',
    'наш': 'наша', 'нашего': 'нашей', 'нашему': 'нашей', 'нашим': 'нашей', 'нашем': 'нашей',
    'ваш': 'ваша', 'вашего': 'вашей', 'вашему': 'вашей', 'вашим': 'вашей', 'вашем': 'вашей',
}
PREP = r'(в|на|о|об|за|к|с|со|у|от|для|без|перед|под|над|между|при|из|над)'

SHORT = {
    'болен': 'больна', 'больна': 'больна',
    'стерилизован': 'стерилизована', 'кастрирован': 'кастрирована',
    'привит': 'привита', 'нагружен': 'нагружена', 'ранен': 'ранена',
    'здоров': 'здорова', 'сыт': 'сыта', 'голоден': 'голодна',
    'спокоен': 'спокойна', 'бодр': 'бодра', 'весел': 'весела',
}

WORD = re.compile(r'[А-Яа-яЁё]+')

def inflect_femn(word):
    """Вернуть женскую форму для глагола прош.вр. или краткого прилаг./причастия, иначе None."""
    for p in morph.parse(word):
        try:
            if p.tag.POS in ('VERB', 'INFN') and 'past' in p.tag:
                nf = p.inflect({'femn'})
                if nf:
                    return nf.word
            elif p.tag.POS in ('ADJS', 'PART') and 'short' in p.tag:
                # краткое прилагательное / краткое причастие
                nf = p.inflect({'femn'})
                if nf:
                    return nf.word
        except Exception:
            pass
    return None

def is_adj_full(word):
    for p in morph.parse(word):
        if p.tag.POS == 'ADJF':
            return True
    return False

def fix_line(line):
    if 'Кудряв' not in line:
        return line
    # 1) местоимения (в любой позиции строки)
    for k, v in PRON.items():
        line = re.sub(r'\b' + k + r'\b', v, line)
    # 2) притяжательные местоимения
    for k, v in POSS.items():
        line = re.sub(r'\b' + k + r'\b', v, line)
    # 3) нём/ним с предлогом
    line = re.sub(r'\b' + PREP + r'\s+нём\b', r'\1 ней', line)
    line = re.sub(r'\b' + PREP + r'\s+ним\b', r'\1 ней', line)

    # токенизируем на слова и разделители
    parts = re.split(r'([А-Яа-яЁё]+)', line)
    # индексы слов
    word_idx = [i for i, p in enumerate(parts) if WORD.fullmatch(p) if p]

    seen = False
    out = list(parts)
    for i, p in enumerate(parts):
        if not WORD.fullmatch(p):
            continue
        if p.startswith('Кудряв'):
            seen = True
            continue
        low = p.lower()
        if low in PRON or low in POSS:
            continue
        # глаголы прош.вр. / краткие прил. ПОСЛЕ клички
        if seen:
            if low in SHORT:
                out[i] = SHORT[low]
                continue
            nf = inflect_femn(p)
            if nf and nf != p:
                out[i] = nf
                continue
        # полное прилагательное ПЕРЕД кличкой
        # смотрим, является ли следующее слово началом 'Кудряв'
        # найдём следующий словесный токен
        nxt = None
        for j in range(i + 1, len(parts)):
            if WORD.fullmatch(parts[j]):
                nxt = parts[j]
                break
        if nxt is not None and nxt.startswith('Кудряв') and is_adj_full(p):
            # инфlect to femn (именительный/винительный по контексту — берём femn форму)
            for pp in morph.parse(p):
                if pp.tag.POS == 'ADJF':
                    nf = pp.inflect({'femn'})
                    if nf:
                        out[i] = nf.word
                        break
    return ''.join(out)

def main():
    total = 0
    for path in sorted(glob.glob(os.path.join(DOCS, '*.txt'))):
        name = os.path.basename(path)
        if not re.match(r'^\d{2}_', name):
            continue
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        new = [fix_line(l) for l in lines]
        changed = sum(1 for a, b in zip(lines, new) if a != b)
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new))
        if changed:
            total += changed
            print(f'{name}: изменено строк={changed}')
    print('ВСЕГО изменено строк:', total)

if __name__ == '__main__':
    main()
