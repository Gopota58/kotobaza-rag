#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Переписывает корпус про рыжего кота так, чтобы он был про Марсика,
вставляя случайные клички в правильных падежах. Делает бэкап исходников."""
import re, os, shutil, glob, random

DOCS = r'C:\rag_project\docs'
BACKUP = r'C:\rag_project\docs_backup'

# Пул кличек: [именительный, родительный/винительный, дательный, творительный, предложный]
NICK = {
    'Марсик':         ['Марсик', 'Марсика', 'Марсику', 'Марсиком', 'Марсике'],
    'Котопузо':       ['Котопузо', 'Котопузы', 'Котопузе', 'Котопузой', 'Котопузе'],
    'Прохиндей':      ['Прохиндей', 'Прохиндея', 'Прохиндею', 'Прохиндеем', 'Прохиндее'],
    'Рыжий властелин': ['Рыжий властелин', 'рыжего властелина', 'рыжему властелину', 'рыжим властелином', 'рыжем властелине'],
    'Рыжий говнюк':   ['Рыжий говнюк', 'рыжего говнюка', 'рыжему говнюку', 'рыжим говнюком', 'рыжем говнюке'],
    'Рыжий обоссун':  ['Рыжий обоссун', 'рыжего обоссуна', 'рыжему обоссуну', 'рыжим обоссуном', 'рыжем обоссуне'],
    'Кудрявая жопа':  ['Кудрявая жопа', 'Кудрявой жопы', 'Кудрявой жопе', 'Кудрявой жопой', 'Кудрявой жопе'],
}
KEYS = list(NICK.keys())

# (regex, индекс падежа). Порядок: сначала составные/длинные, потом короткие.
PATTERNS = [
    (re.compile(r'рыжего кота'), 1),
    (re.compile(r'рыжему коту'), 2),
    (re.compile(r'рыжим котом'), 3),
    (re.compile(r'рыжем коте'), 4),
    (re.compile(r'рыжий кот'), 0),
    (re.compile(r'котёнка'), 1),
    (re.compile(r'котёнку'), 2),
    (re.compile(r'котёнком'), 3),
    (re.compile(r'котёнке'), 4),
    (re.compile(r'котёнок'), 0),
    (re.compile(r'\bкота\b'), 1),
    (re.compile(r'\bкоту\b'), 2),
    (re.compile(r'\bкотом\b'), 3),
    (re.compile(r'\bкоте\b'), 4),
    (re.compile(r'\bкот\b'), 0),
]

def make_picker():
    used = set()
    def pick(ci):
        if len(used) < len(KEYS):
            choices = [k for k in KEYS if k not in used]
            k = random.choice(choices)
        else:
            k = random.choice(KEYS)
        used.add(k)
        return NICK[k][ci]
    return pick, used

def transform_line(line, pick):
    if line.startswith('ИНСТРУКЦИИ:'):
        line = line.replace('РЫЖЕГО КОТА', 'МАРСИКА')
        line = line.replace('рыжего кота', 'Марсика')
        line = line.replace('КОТА В КВАРТИРЕ', 'МАРСИКА В КВАРТИРЕ')
        line = re.sub(r'\bКОТА\b', 'МАРСИКА', line)
    for rx, ci in PATTERNS:
        line = rx.sub(lambda m, ci=ci: pick(ci), line)
    return line

def main():
    os.makedirs(BACKUP, exist_ok=True)
    files = sorted(glob.glob(os.path.join(DOCS, '*.txt')))
    total = 0
    for path in files:
        name = os.path.basename(path)
        if not re.match(r'^\d{2}_', name):
            continue  # пропускаем MEMORY.md и прочее
        shutil.copy2(path, os.path.join(BACKUP, name))
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        pick, used = make_picker()
        before = ' '.join(lines)
        new_lines = [transform_line(l, pick) for l in lines]
        after = ' '.join(new_lines)
        # грубый подсчёт замен: число вхождений кличек
        cnt = sum(after.count(k) for k in KEYS)
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))
        print(f'{name}: вхождений кличек={cnt}, уникальных в файле={len(used)}/7')
        total += cnt
    print('ВСЕГО вхождений кличек:', total)

if __name__ == '__main__':
    main()
