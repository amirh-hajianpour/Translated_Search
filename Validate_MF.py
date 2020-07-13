import sys
import re

class Validate_MF:
    def Run(annot_file):
        annot_string = open(annot_file, 'r')
        seq = annot_string.read()
        #sequence = re.sub(r'^;.*|!|^>.*|^$', '', seq)
        last_idx = 0
        for index, line in enumerate(seq.split('\n')):
            line = re.sub(' +', ' ', line.strip()).split(' ')
            if len(line) != 2:
                print(index, line)
                raise ValueError('Invalid line.')
            if last_idx != 0 and int(line[0]) != int(last_idx) + len(last_line):
                print(index, int(line[0]), int(last_idx) + len(last_line))
            last_idx = line[0]
            last_line = line[1]
        le = 1
        for m in re.findall(r'^\s*(\d+)\s*(.+)$', seq, re.MULTILINE):
            if le != int(m[0]):
                print(m, le)
                le = int(m[0])
            # le += len([e for e in m[1] if e != '!'])
            le += len(m[1])

Validate_MF.Run(sys.argv[1])
