import sys
import subprocess
import os
import re


class Translated_Search:
    dna = ""
    genetic_code = {
        # Phe
        'UUU': 'Phe', 'UUC': 'Phe',
        # Leu
        'UUA': 'Leu', 'UUG': 'Leu', 'CUU': 'Leu', 'CUC': 'Leu', 'CUA': 'Leu', 'CUG': 'Leu',
        # Ser
        'UCU': 'Ser', 'UCC': 'Ser', 'UCA': 'Ser', 'UCG': 'Ser', 'AGU': 'Ser', 'AGC': 'Ser',
        # Tyr
        'UAU': 'Tyr', 'UAC': 'Tyr',
        # STOP
        'UAA': 'STOP', 'UAG': 'STOP', 'UGA': 'STOP',
        # Cys
        'UGU': 'Cys', 'UGC': 'Cys',
        # Trp
        'UGG': 'Trp',
        # Pro
        'CCU': 'Pro', 'CCC': 'Pro', 'CCA': 'Pro', 'CCG': 'Pro',
        # His
        'CAU': 'His', 'CAC': 'His',
        # Gln
        'CAA': 'Gln', 'CAG': 'Gln',
        # Arg
        'CGU': 'Arg', 'CGC': 'Arg', 'CGA': 'Arg', 'CGG': 'Arg', 'AGA': 'Arg', 'AGG': 'Arg',
        # Ile
        'AUU': 'Ile', 'AUC': 'Ile', 'AUA': 'Ile',
        # Met
        'AUG': 'Met',
        # Thr
        'ACU': 'Thr', 'ACC': 'Thr', 'ACA': 'Thr', 'ACG': 'Thr',
        # Asn
        'AAU': 'Asn', 'AAC': 'Asn',
        # Lys
        'AAA': 'Lys', 'AAG': 'Lys',
        # Val
        'GUU': 'Val', 'GUC': 'Val', 'GUA': 'Val', 'GUG': 'Val',
        # Ala
        'GCU': 'Ala', 'GCC': 'Ala', 'GCA': 'Ala', 'GCG': 'Ala',
        # Asp
        'GAU': 'Asp', 'GAC': 'Asp',
        # Glu
        'GAA': 'Glu', 'GAG': 'Glu',
        # Gly
        'GGU': 'Gly', 'GGC': 'Gly', 'GGA': 'Gly', 'GGG': 'Gly'
    }
    abbreviation = {
        'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C', 'Glu': 'E', 'Gln': 'Q',
        'Gly': 'G', 'His': 'H', 'Ile': 'I', 'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F',
        'Pro': 'P', 'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V', 'STOP': '*'
    }
    hmms_coordinates = []
    hits_coordinates = []

    def __init__(self):
        print()

    @staticmethod
    # Makes a complementary DNA strand
    def complementor(dna):
        dna = dna.upper()

        # T => A, and A => T
        dna = dna.replace('A', '1')
        dna = dna.replace('T', 'A')
        dna = dna.replace('1', 'T')

        # G => C, and C => G
        dna = dna.replace('C', '1')
        dna = dna.replace('G', 'C')
        dna = dna.replace('1', 'G')

        return dna

    @staticmethod
    # Transcribes and Translates a DNA string
    # Ignores remaining residue(s), skips abnormal codons
    def translator(dna, frame_number = 0, reverse = False, stop_codon = '*'):
        dna = dna.upper()
        dna = dna[frame_number:]

        # Changing STOP codon translating symbol
        Translated_Search.abbreviation.update(STOP = str(stop_codon))

        # Transcription (T => U)
        transcribed_dna = dna.replace('T', 'U')

        # Translation (codon => amino acid)
        start = end = 0 # codon coordinates
        translated_dna = ""
        # The remaining residue(s) is/are ignored (e.g. AAAU => AAA = K)
        while (len(transcribed_dna) - end) / 3 >= 1:
            end = start + 3
            codon = transcribed_dna[start:end]
            # If reverse ORF, codons should be read backward
            if reverse:
                codon = codon[::-1]
            # Abnormal codons are completely skipped. (e.g. AAUPUU => AAU = N)
            if str(codon) in Translated_Search.genetic_code.keys():
                amino_acid = Translated_Search.genetic_code[str(codon)]
                translated_dna += Translated_Search.abbreviation[amino_acid]
            start = end
        return translated_dna

    @staticmethod
    # Converts a string to FASTA format
    def string_to_fasta(sequence, title = "", line_length = 80):
        counter = 1
        fasta = ""
        for char in str(sequence):
            if (counter % line_length) == 0:
                fasta += char + '\n'
            else:
                fasta += char
            counter += 1
        if title != "":
            fasta = '>' + title + '\n' + fasta + '\n'
        return fasta

    @staticmethod
    # Converts a FASTA format to string
    def fasta_to_string(fasta):
        lines = fasta.split('\n')
        current_line_index = 0

        # Ignoring the comments
        while lines[current_line_index].startswith(';'):
            current_line_index += 1

        # Getting the title line
        while not lines[current_line_index].startswith('>'):
            current_line_index += 1
        title = lines[current_line_index].replace('>', '')

        # Getting the whole sequence as a one-line string
        lines = lines[current_line_index+1:len(lines)]
        sequence = ""
        for line in lines:
            sequence += line
        sequence = sequence.replace('\n', '')

        return title, sequence

    # Computes a score for a set of hits considering:
    # overlap as penalty, and match as positive score
    def scoring_function(self, current):
        current = list(map(list, current))
        overlap = 0
        if len(current) == 1:
            return current[0][1] - current[0][0]

        for i in range(0, len(current)):
            for j in range(i + 1, len(current)):
                if current[i][1] <= current[j][0] or current[i][0] >= current[j][1]:
                    if current[i][1] <= current[j][0]:
                        overlap += current[j][0] - current[i][1] + 1
                    else:
                        overlap += current[i][0] - current[j][1] + 1
                elif current[i][0] <= current[j][0] and current[i][1] <= current[j][1]:
                    overlap += current[i][1] - current[j][0] + 1
                elif current[i][0] >= current[j][0] and current[i][1] <= current[j][1]:
                    overlap += current[i][1] - current[i][0]
                elif current[i][0] >= current[j][0] and current[i][1] >= current[j][1]:
                    overlap += current[j][1] - current[i][0] + 1
                elif current[i][0] <= current[j][0] and current[i][1] >= current[j][1]:
                    overlap += current[j][1] - current[j][0]
                else:
                    print("UNKNOWN EXCEPTION!")

        coverage = 0
        for i in range(0, len(current)):
            coverage += current[i][1] - current[i][0]

        return coverage - (2*overlap)

    # Creates a binary tree in which a hit is selected or not: 2^hits
    def naive(self, hits, current, n):
        # Base Case
        if n == 0:
            return self.scoring_function(current), current

        # return the maximum of two cases:
        # (1) n-1 th item not included
        # (2) n-1 th item included
        old_current = current.copy()
        current.add(hits[n - 1])
        left_score, left_set = self.naive(hits, current, n - 1)
        right_score, right_set = self.naive(hits, old_current, n - 1)

        if left_score > right_score:
            return left_score, left_set
        else:
            return right_score, right_set

    # Checks if an exon is found in the search and returns it in addition to the overlap
    def is_hit(self, exon_start, exon_end):
        for i in range(0, len(self.hits_coordinates)):
            temp = self.hits_coordinates[i].copy()
            # Going from Amino Acid coordinates to Nucleic Acid coordinates
            temp[0] = (temp[0] - 1) * 3 + 1
            temp[1] = temp[1] * 3
            # if they have overlap
            if max(exon_start, temp[0]) < min(exon_end, temp[1]):
                return True, min(exon_end, temp[1]) - max(exon_start, temp[0]) + 1
        return False, 0

    # Extracts exons of a gene in a master file format (indices ('i') and sequences('s'))
    def mf_exon_reader(self, mf_file, gene, result):
        # Reading the MasfterFile File
        mf = open(mf_file, 'r')
        lines = mf.read().split('\n')

        # Finding the starting line of the FASTA file (<)
        current_line_index = 0
        current_line = re.sub(' +', ' ', lines[current_line_index]).split(' ')
        while not current_line[0].startswith('>'):
            current_line_index = current_line_index + 1
            current_line = re.sub(' +', ' ', lines[current_line_index]).split(' ')

        # Finding the starting line of the gene
        while current_line[0] != ';' or current_line[1] != ('G-' + gene) or current_line[3] != 'start':
            current_line_index = current_line_index + 1
            current_line = re.sub(' +', ' ', lines[current_line_index]).split(' ')

        is_exon = False # it's an exon sequence line
        exon_number = -1    # current number of the exon
        exon_starts = [0] * 100 # starting indices of exons
        exon_ends = [0] * 100   # ending indices of exons
        exon = [""] * 100   # list of sequences of exons
        temp_exon = [""] * 100  # buffer exon
        exon_start_line = 0 # starting line of the exon

        # Looping until reaching end of the gene
        while current_line[0] != ';' or current_line[1] != ('G-' + gene) or current_line[3] != 'end':
            # exon starts
            if current_line[0] == ';' and current_line[1].startswith('G-' + gene + '-E') and current_line[3] == 'start':
                is_exon = True
                exon_number = exon_number + 1
                # Looking for the start index of the current exon
                temp_line_index = current_line_index
                temp_line = current_line
                while temp_line[0].startswith(';'):
                    temp_line_index = temp_line_index + 1
                    temp_line = re.sub(' +', ' ', lines[temp_line_index]).strip().split(' ')
                exon_starts[exon_number] = int(temp_line[0])
                exon_start_line = temp_line_index
            # exon ends
            elif current_line[0] == ';' and current_line[1].startswith('G-' + gene + '-E') and current_line[3] == 'end':
                is_exon = False
                # Copying buffer exon to list of sequences of exons
                for i in range(0, current_line_index - exon_start_line):
                    exon[exon_number] = exon[exon_number] + temp_exon[i]
                temp_exon = [""] * 100
                # Looking for the end index of the current exon
                temp_line_index = current_line_index
                temp_line = current_line
                while temp_line[0].startswith(';'):
                    temp_line_index = temp_line_index + 1
                    temp_line = re.sub(' +', ' ', lines[temp_line_index]).strip().split(' ')
                exon_ends[exon_number] = int(temp_line[0]) - 1
            current_line_index = current_line_index + 1
            current_line = re.sub(' +', ' ', lines[current_line_index]).strip().split(' ')
            # Filling buffer exon
            if is_exon and not current_line[0].startswith(';'):
                temp_exon[current_line_index - exon_start_line] = temp_exon[current_line_index - exon_start_line] + current_line[1]
        # Printing result
        if result != 's':
            print("\nPositions of \"" + gene + "\" exons in the annotated file:")
            print("------------------------------------------------")
            print('{0} {1:>6} {2:>5} {3:>8} {4:>4} {5:>8} {6:>9}'.format("Exon", "Start", "End", "Length", "Hit", "Overlap", "Accuracy"))
            print('{0} {1:>6} {2:>6} {3:>7} {4:>4} {5:>8} {6:>9}'.format("----", "-----", "-----", "------", "---", "-------", "--------"))
            yes_no = lambda x: 'Yes' if x == True else 'No'
            exon_length = 0
            overlap_length = 0
            number_of_hits = 0
            for i in range(0, exon_number + 1):
                if self.is_hit(exon_starts[i], exon_ends[i])[0]:
                    number_of_hits += 1
                exon_length += exon_ends[i] - exon_starts[i] + 1
                overlap_length += self.is_hit(exon_starts[i], exon_ends[i])[1]
                print('#' + '{0:>3} {1:>6} {2:>6} {3:>7} {4:>4} {5:>8} {6:>9.2f}'.format(i+1, exon_starts[i], exon_ends[i] \
                    , exon_ends[i] - exon_starts[i] + 1, yes_no(self.is_hit(exon_starts[i], exon_ends[i])[0]) \
                    , self.is_hit(exon_starts[i], exon_ends[i])[1], self.is_hit(exon_starts[i], exon_ends[i])[1] / (exon_ends[i] - exon_starts[i] + 1)))
            print("\nOverall annotation statistics summary:")
            print("--------------------------------------")
            print('{0:<40} {1:>3} {2:<13}'.format("Total spliced RNA size:", exon_length, "nucleotides"))
            print('{0:<40} {1:>3} {2:<13}'.format("Total overlap length:", overlap_length, "nucleotides"))
            print('{0:<40} {1:>4} {2:}'.format("Number of found exons (TPR):", number_of_hits, str("out of " + str(exon_number + 1) + ' = ' + format(number_of_hits/(exon_number + 1),'.2f'))))
            print('{0:<40} {1:>3}'.format("Overall score:", format(overlap_length/exon_length, '.2f')))
        if result != 'i':
            print()
            print("Sequences of \"" + gene + "\" exons in the annotated file:")
            print("----------------------------------------------")
            for i in range(0, exon_number + 1):
                print('{0:>0} {1:>2} {2:>0} {3:>3}'.format("Exon #", str(i+1), " ", exon[i]))
        print()
        return exon_starts, exon_ends

    # Reads HMMER output and extracts necessary information
    def hmmer_reader(self, output):
        line_counter = 0
        current_hit = 0
        number_of_hits = 0
        lines = output.split('\n')
        for line in lines:
            reformed_line = re.sub(' +', ' ', line).strip().split(' ')
            # Finding the number of hits
            if reformed_line[0] == 'Scores' and reformed_line[1] == 'for' and reformed_line[2] == 'complete' and reformed_line[3] == 'sequences':
                read = line_counter + 4
                if len(re.sub(' +', ' ', lines[read]).strip().split(' ')) < 9:
                    return "No hit found!", "No hit found!"
                while re.sub(' +', ' ', lines[read]).strip().split(' ')[0] != "Domain":
                    if len(re.sub(' +', ' ', lines[read]).strip().split(' ')) >= 9:
                        number_of_hits += int(re.sub(' +', ' ', lines[read]).strip().split(' ')[7])
                    read += 1
                self.hmms_coordinates = list(range(0, number_of_hits))
                self.hits_coordinates = list(range(0, number_of_hits))
            # Finding hits in each sequence
            if reformed_line[0] == '>>':
                read = line_counter + 3
                # Reading the start and the end index of each hit
                while len(re.sub(' +', ' ', lines[read]).strip().split(' ')) == 16:
                    self.hmms_coordinates[current_hit] = list(map(int, re.sub(' +', ' ', lines[read]).strip().split(' ')[6:8]))
                    self.hits_coordinates[current_hit] = list(map(int, re.sub(' +', ' ', lines[read]).strip().split(' ')[9:11]))
                    read += 1
                    current_hit += 1
            line_counter += 1
        return self.hmms_coordinates, self.hits_coordinates

    def hmmsearch(self, hmm_profile, dna_seq):
        # Reading HMM protein profile and the DNA sequence as arguments
        print("Reading the hmm file ...")
        protein_profile_file = open(hmm_profile, 'r')

        print("Reading the dna file ...")
        dna_sequence_file = open(dna_seq, 'r')

        Translated_Search.dna = Translated_Search.fasta_to_string(dna_sequence_file.read())

        # Creating the translated_dna_file with 6 different sequences (ORFs) with 6 different recognizable titles
        print("\nForward ORFs:")
        print("-------------")
        translated_dna = ""
        for frame_number in range(3):
            print("Translating frame number ", frame_number, " ...")
            translated_frame = Translated_Search.translator(Translated_Search.dna[1], frame_number)
            translated_dna = translated_dna + Translated_Search.string_to_fasta(translated_frame, title = ('_' + str(frame_number) + Translated_Search.dna[0]))

        print("\nReverse ORFs:")
        print("-------------")
        for frame_number in range(3):
            print("Translating frame number ", frame_number, " ...")
            translated_frame = Translated_Search.translator(Translated_Search.complementor(Translated_Search.dna[1]), frame_number, True)
            translated_dna = translated_dna + Translated_Search.string_to_fasta(translated_frame, title = ('_r' + str(frame_number) + Translated_Search.dna[0]))

        # Writing the translated_dna_file to a file for HMMER use
        translated_frames_file = open("translated_frames.fa", 'w+')
        translated_frames_file.write(translated_dna)
        seq_path = os.path.abspath("translated_frames.fa")
        translated_frames_file.seek(0, 0)

        # "--nobias", "--incE", "10", "--incdomE", "10", "--F3", "1", "--nonull2", "--max"
        print("\nRunning HMMER ...")
        process = subprocess.run(["hmmsearch", "--F3", "1", hmm_profile, seq_path], \
            stdout=subprocess.PIPE, universal_newlines=True)
        output = process.stdout
        output_file = open("hmm_output.txt", 'w+')
        output_file.write(output)
        print(output)
        return self.hmmer_reader(output)

    def run(self, dna_seq, hmm_profile, annotated, gene, format):
        hs = self.hmmsearch(hmm_profile, dna_seq)
        ex = self.mf_exon_reader(annotated, gene, format)


t = Translated_Search()
if sys.argv[1] == '-h':
    print("python3 Translation.py dna_seq = "", hmm_profile = "", annotated = "", gene = "", format = """)
else:
    my_output = t.run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])


# if hmms_coordinates != "No hit found!":
#     my_out = t.naive(list(map(tuple, hmms_coordinates)), set(), len(hmms_coordinates))
#     print("Highest score is: ", my_out[0], ", and the best set is: ", my_out[1])
# else:
#     print(hmms_coordinates)
