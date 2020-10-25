import sys
import re

class Annot_To_FASTA:

    @staticmethod
    def run(annot_file, fasta_length = 60, write = True):
        annot_string = open(annot_file, 'r')
        file = annot_string.read()

        fasta = ''
        for index, line in enumerate(file.split('\n')):
            if line.startswith('>'):
                fasta = line + '\n'
                break

        sequence = re.sub(r'^\s*\d+\s*|^;.*|[! \t]|^>.*', '', file, flags=re.M).upper()
        residue = ''
        for line in sequence.split('\n'):
            if line.strip() != '':
                residue += line
                if len(residue) == fasta_length:
                    fasta += residue + '\n'
                    residue = ''
                elif len(residue) > fasta_length:
                    fasta += residue[0:fasta_length] + '\n'
                    residue = residue[fasta_length:]

        if len(residue) != 0:
            fasta += residue + '\n'

        if write:
            fasta_file = open(annot_string.name.split('.')[0]+'.fasta2', 'w+')
            fasta_file.write(fasta)

        return fasta

Annot_To_FASTA.run(sys.argv[1])
