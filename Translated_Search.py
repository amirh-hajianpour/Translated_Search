import sys
import subprocess
import os
import re
import math
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
from scipy.stats import entropy
from scipy.spatial import distance_matrix
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

class Translated_Search:

    genetic_code_uni = {
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
    genetic_code_vmit = {
        # Phe
        'UUU': 'Phe', 'UUC': 'Phe',
        # Leu
        'UUA': 'Leu', 'UUG': 'Leu', 'CUU': 'Leu', 'CUC': 'Leu', 'CUA': 'Leu', 'CUG': 'Leu',
        # Ile
        'AUU': 'Ile', 'AUC': 'Ile',
        # Met
        'AUA': 'Met', 'AUG': 'Met',
        # Val
        'GUU': 'Val', 'GUC': 'Val', 'GUA': 'Val', 'GUG': 'Val',
        # Ser I
        'UCU': 'Ser', 'UCC': 'Ser', 'UCA': 'Ser', 'UCG': 'Ser',
        # Pro
        'CCU': 'Pro', 'CCC': 'Pro', 'CCA': 'Pro', 'CCG': 'Pro',
        # Thr
        'ACU': 'Thr', 'ACC': 'Thr', 'ACA': 'Thr', 'ACG': 'Thr',
        # Ala
        'GCU': 'Ala', 'GCC': 'Ala', 'GCA': 'Ala', 'GCG': 'Ala',
        # Tyr
        'UAU': 'Tyr', 'UAC': 'Tyr',
        # STOP I
        'UAA': 'STOP', 'UAG': 'STOP',
        # His
        'CAU': 'His', 'CAC': 'His',
        # Gln
        'CAA': 'Gln', 'CAG': 'Gln',
        # Asn
        'AAU': 'Asn', 'AAC': 'Asn',
        # Lys
        'AAA': 'Lys', 'AAG': 'Lys',
        # Asp
        'GAU': 'Asp', 'GAC': 'Asp',
        # Glu
        'GAA': 'Glu', 'GAG': 'Glu',
        # Cys
        'UGU': 'Cys', 'UGC': 'Cys',
        # Trp
        'UGA': 'Trp', 'UGG': 'Trp',
        # Arg
        'CGU': 'Arg', 'CGC': 'Arg', 'CGA': 'Arg', 'CGG': 'Arg',
        # Ser II
        'AGU': 'Ser', 'AGC': 'Ser',
        # STOP II
        'AGA': 'STOP', 'AGG': 'STOP',
        # Gly
        'GGU': 'Gly', 'GGC': 'Gly', 'GGA': 'Gly', 'GGG': 'Gly'
    }
    abbreviation = {
        'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C', 'Glu': 'E', 'Gln': 'Q',
        'Gly': 'G', 'His': 'H', 'Ile': 'I', 'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F',
        'Pro': 'P', 'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V', 'STOP': '*'
    }

    def __init__(self):
        self.dna = ''
        self.annotation = ''
        self.gene_name = 2*[0]

        self.hmms_coordinates = []
        self.hits_coordinates = []
        self.hits_frame_number = []

        self.exons = []
        self.exon_starts = []
        self.exon_ends = []

        self.annot_genes = []
        self.genome_codon_frequencies = []
        self.hits_codon_frequencies = []

        self.FPs = []
        self.message = '' # TODO: remove this
        self.output = ''

    # Takes a position in amino acid coordinates (translated DNA) to nucleic acid coordinates (DNA)
    # Returns a nucleic acid position
    def amino_to_nucleic(self, aa_pos, frame_number):
        if aa_pos <= 0:
            raise ValueError('Negative amino acid position.')
        if frame_number not in [1,2,3]:
            raise ValueError('Frame number is not 1, 2, or 3.')

        start = aa_pos * 3 - 2 + frame_number - 1
        end = aa_pos * 3 + frame_number - 1

        return start, end

    # Takes a position in nucleic acid coordinates (DNA) to amino acid coordinates (translated DNA)
    # Returns an amino acid position
    def nucleic_to_amino(self, na_pos, frame_number):
        if na_pos <= 0:
            raise ValueError('Negative amino acid position.')
        if frame_number not in [1,2,3]:
            raise ValueError('Frame number is not 1, 2, or 3.')
        if na_pos - frame_number < 0:
            raise ValueError('Nucleic acid position: ' + str(na_pos) + ', or frame number: ' + str(frame_number) + ' has invalid value.')

        return (na_pos - frame_number + 1) // 3 + 1

    # Detects whether a sequence is DNA, RNA, Protein, or invalid
    # Returns the type of the sequence
    def seq_type(self, sequence, spec_chars = []):
        if not sequence:
            raise ValueError('Sequence has length 0.')

        sequence = sequence.upper()

        dna = 'ATCG'
        rna = 'AUCG'
        protein = 'ACDEFGHIKLMNPQRSTVWY'

        if not re.findall('[^'+dna+''.join(spec_chars)+']', sequence):
            return 'DNA'
        elif not re.findall('[^'+rna+''.join(spec_chars)+']', sequence):
            return 'RNA'
        elif not re.findall('[^'+protein+''.join(spec_chars)+']', sequence):
            return 'Protein'
        else:
            raise ValueError('Sequence is invalid.')

    # Makes a reverse, complement DNA strand
    # Returns the reversed, complement DNA string
    def rev_comp(self, dna):
        if not dna:
            raise ValueError('DNA has length 0.')

        dna = dna.upper()
        c_dna = dna

        # T => A, and A => T
        c_dna = c_dna.replace('A', '~')
        c_dna = c_dna.replace('T', 'A')
        c_dna = c_dna.replace('~', 'T')

        # G => C, and C => G
        c_dna = c_dna.replace('C', '~')
        c_dna = c_dna.replace('G', 'C')
        c_dna = c_dna.replace('~', 'G')

        return c_dna[::-1]

    # Transcribes and Translates a DNA string
    # Returns the translated DNA, list of abnormal codons, and whether the length of the DNA was divisible by 3 or not
    def translate(self, my_dna, frame_number = 1, reverse = False, stop_codon = '*', ab_codon = 'X'):
        if not my_dna:
            raise ValueError('DNA has length 0.')
        if frame_number not in [1,2,3]:
            raise ValueError('Frame number is not 1, 2, or 3.')
        if len(stop_codon) > 1:
            raise ValueError('Stop-codon has invalid value.')

        my_dna = my_dna.upper()

        # Changing STOP codon translation symbol
        self.abbreviation.update(STOP = stop_codon)

        # If reverse ORF, codons should be read from reverse complement
        transcribed_dna = my_dna if not reverse else self.rev_comp(my_dna)

        transcribed_dna = transcribed_dna[frame_number-1:]

        # Transcription (T => U)
        transcribed_dna = transcribed_dna.replace('T', 'U')

        # Translation (codon => amino acid)
        start = end = 0 # codon coordinates
        translated_dna = ""
        abnormal_codons = []

        # The remaining residue(s) is/are ignored (e.g. AAAU => AAA = K)
        while (len(transcribed_dna) - end) >= 3:
            end = start + 3
            codon = transcribed_dna[start:end]

            # Abnormal codons are translated to ab_codon and stored. (e.g. AAUPUU => NX)
            if codon in self.genetic_code_vmit.keys():
                amino_acid = self.genetic_code_vmit[codon]
                translated_dna += self.abbreviation[amino_acid]
            else:
                abnormal_codons.append(codon)
                translated_dna += ab_codon
            start = end

        return translated_dna, abnormal_codons, len(transcribed_dna) % 3

    # Makes a FASTA string from a sequence
    # Returns a FASTA string with specified title and length
    def string_to_fasta(self, sequence, title = "", line_length = 80):
        if not sequence:
            raise ValueError('Sequence has length 0.')
        if line_length < 1:
            raise ValueError('Line length is invalid.')

        sequence = sequence.upper()
        sequence = sequence.replace('\n','')
        title = title.replace('\n', '')

        fasta = ""
        start = end = 0
        # Making each line of the FASTA Format
        while len(sequence) - end >= line_length:
            end = start + line_length
            fasta += sequence[start:end] + '\n'
            start = end

        # Adding last line
        if len(sequence) - end > 0:
            fasta += sequence[start:] + '\n'

        # Adding title as first line
        if title:
            fasta = '>' + title + '\n' + fasta + '\n'
        else:
            fasta = '>' + 'Unknown' + '\n' + fasta + '\n'

        return fasta

    # Extracts the sequence of a FASTA string
    # Returns the sequence and the title
    def fasta_to_string(self, fasta):
        if not fasta:
            raise ValueError('Fasta has length 0.')

        # Getting the title line
        index = 0
        for index, line in enumerate(fasta.split('\n')):
            if line.startswith('>'):
                title = line.replace('>', '')
                break

        sequence = ""
        # Getting the whole sequence as a one-line string
        for line in fasta.split('\n')[index+1:]:
            sequence += line

        return sequence, title

    # TODO: Crossing detection report should be implemented in a better way
    # Computes a score for a set of hmms considering:
    # (1) overlap as penalty, (2) match (coverage) as positive score
    # Rejects sets of hmms that have crossings
    def score(self, hmms_set):
        hmms = list(map(list, hmms_set))

        # If the set only contains one hit
        if len(hmms) == 1:
            return hmms[0][1] - hmms[0][0] + 1

        # Checking for crossing (invalid order of) hits
        hmms = sorted(hmms, key=lambda tup: tup[0])
        hmm_indices = []

        for hmm in hmms:
            hmm_indices.append(self.hmms_coordinates.index(hmm))

        hits = [self.hits_coordinates[i] for i in hmm_indices]
        if hits != sorted(hits, key=lambda tup: tup[0]):
            self.message = "Crossing detected!"
            return -1000000

        # Looping through all pairs of hits to calculate the overall overlap
        overlap = 0
        for i in range(len(hmms)):
            for j in range(i + 1, len(hmms)):
                if max(hmms[i][0], hmms[j][0]) < min(hmms[i][1], hmms[j][1]):
                    overlap += min(hmms[i][1], hmms[j][1]) - max(hmms[i][0], hmms[j][0]) + 1

        # Calculating the coverage (ovrelap is being added 2 times)
        coverage = 0
        for i in range(len(hmms)):
            coverage += hmms[i][1] - hmms[i][0] + 1

        return coverage - (2*overlap)

    # Goes through all combinations of a set of hits: 2^hits
    def combinations(self, hits, bag_of_hits, n):
        # Base Case
        if n == 0:
            return self.score(bag_of_hits), bag_of_hits

        # Keeping the bag unchanged
        old_bag_of_hits = bag_of_hits.copy()
        # Adding the last element of the hits list to the bag
        bag_of_hits.add(hits[n - 1])
        # Calculating the score of the bag if n-1th hit was added
        left_score, left_set = self.combinations(hits, bag_of_hits, n - 1)
        # Calculating the score of the bag if n-1th hit was not added
        right_score, right_set = self.combinations(hits, old_bag_of_hits, n - 1)

        # Keeping the item if it led to a better score
        if left_score >= right_score:
            return left_score, left_set
        # Dropping the item if it didn't lead to a better score
        else:
            return right_score, right_set

    # TODO: Maybe remove the hit reference to make this method to be a general overlap finder method
    # Checks if an exon is found in the search
    # Returns the indices of the corresponding hit(s)
    def is_hit(self, exon_start, exon_end):
        if exon_start < 0:
            raise ValueError('Starting position of exon is a negative value.')
        if exon_end <= 0:
            raise ValueError('Ending position of exon is a negative value.')
        if exon_start >= exon_end:
            raise ValueError('Ending position of exon is lower that its ending position.')

        hits_indices = []
        for i, hit in enumerate(self.hits_coordinates):
            # if they have overlap
            if max(exon_start, min(self.amino_to_nucleic(hit[0], self.hits_frame_number[i]))) \
                <= min(exon_end, max(self.amino_to_nucleic(hit[1], self.hits_frame_number[i]))):
                hits_indices.append(i)

        return hits_indices

    # Extracts exons of a gene in a master file format
    def exons_finder(self):
        if not self.gene_name[0]:
            raise ValueError('Gene name is invalid.')

        # Reading the annotation string line by line
        lines = self.annotation.split('\n')

        # Finding the starting line of the FASTA file (>)
        current_line_index = 0
        while not lines[current_line_index].strip().startswith('>'):
            current_line_index += 1

        current_line = re.sub(' +', ' ', lines[current_line_index].strip()).split(' ')
        # Finding the starting line of the gene
        while current_line[0] != ';' or len(re.findall(('^G-'+self.gene_name[0]+'_{0,1}\d{0,1}$'), current_line[1])) == 0 or not current_line[3].startswith('start'):
            current_line_index += 1
            current_line = re.sub(' +', ' ', lines[current_line_index].strip()).split(' ')
        multi_part = True if len(re.findall(('^G-'+self.gene_name[0]+'_\d$'), current_line[1])) > 0 else False # to handle multi-parted exons: G-cob_1, G-cob_2
        is_exon = False # it's an exon sequence line
        new_start = False # to handle multi-parted exons: G-cob_1, G-cob_2
        uncounted = 0 # counting prev seqs that didnt have indices for calculating starting position of the exon
        temp_exon = ''  # buffer exon
        # Reading all the exons of the gene
        while current_line_index < len(lines) and (current_line[0] != ';' or len(re.findall(('^G-'+self.gene_name[0]+'_{0,1}\d{0,1}$'), current_line[1])) == 0 or \
            current_line[3] != 'end' or multi_part):
            current_line = re.sub(' +', ' ', lines[current_line_index]).strip().split(' ')
            # Filling buffer exon
            if is_exon and not current_line[0].startswith(';') and current_line != ['']:
                # Looking for the start index of the current exon
                if len(current_line) > 1:
                    if current_line[0].isnumeric():
                        if not new_start:
                            self.exon_starts.append(int(current_line[0]) - uncounted)
                            new_start = True
                        temp_exon += ''.join(current_line[1:])
                    else:
                        temp_exon += ''.join(current_line)
                        uncounted += len(''.join(current_line))
                elif len(current_line) == 1:
                    temp_exon += current_line[0]
                    uncounted += len(current_line[0])

            # exon starts
            elif current_line[0] == ';' and len(re.findall(('^G-'+self.gene_name[0]+'.*-E'), current_line[1])) > 0 and current_line[3].startswith('start'):
                is_exon = True
                new_start = False
                uncounted = 0
            # exon ends
            elif current_line[0] == ';' and len(re.findall(('^G-'+self.gene_name[0]+'.*-E'), current_line[1])) > 0 and current_line[3].startswith('end'):
                is_exon = False
                # Copying buffer exon to list of sequences of exons
                self.exons.append(temp_exon)
                # Looking for the end index of the current exon
                counter = 1
                uncounted2 = 0
                while True:
                    if re.sub(' +', ' ', lines[current_line_index + counter]).strip().split(' ')[0].isnumeric():
                        break
                    else:
                        if not lines[current_line_index + counter].strip().startswith(';'):
                            uncounted2 += len(re.sub(' +', '', lines[current_line_index + counter]))
                    counter += 1
                end_index = int(re.sub(' +', ' ', lines[current_line_index + counter]).strip().split(' ')[0])-1-uncounted2
                if len(self.exon_starts) <= len(self.exon_ends):
                    self.exon_starts.append(end_index-len(temp_exon))
                self.exon_ends.append(end_index)
                temp_exon = ''

            current_line_index += 1
        return self.exon_starts, self.exon_ends

    # Checks if all indices of the annotation are valid.
    def annot_valid(self):
        valid = True
        lines = self.annotation.split('\n')
        prev_index = new_index = 1
        prev_seq = new_seq = ''
        for idx, line in enumerate(lines):
            line = re.sub(' +', ' ', line.strip()).split(' ')
            if not line[0].startswith('>') and not line[0].startswith(';') and line != ['']:
                if line[0].isnumeric():
                    new_index = line[0]
                    new_seq = ''.join(line[1:])
                else:
                    new_index = int(prev_index) + len(prev_seq)
                    new_seq = ''.join(line)
                if int(prev_index) + len(prev_seq) != int(new_index):
                    if int(prev_index) + len(re.sub('!', '', prev_seq)) != int(new_index):
                        if valid:
                            print('\n***Incorrect index in annotation:***\n')
                            valid = False
                        print(line)
                prev_index = new_index
                prev_seq = new_seq
        return valid

    # Prints some information about the annotation (result: indices ('i') and sequences('s'))
    def annot_info(self, result = 'i'):
        # Printing result
        if result != 's':
            print()
            print("Exons and HMMER Hits for gene: '", self.gene_name[0], "'\n", '-'*(len("Exons and HMMER Hits for gene: '" + self.gene_name[0] + "'")), sep = '')
            print('{0} {1:>8} {2:>11} {3:>9} {4:>9} {5:>4} {6:>8} {7:>10} {8:>8} {9:>9} {10:>10} {11:>6} {12:>5} {13:>8}'.format("Exon", "E_Frame", "Exon_Start" \
            , "Exon_End", "E-Length", "Hit", "H_Frame", "Hit_Start", "Hit_End", "H-Length", "H-Overlap", " TPR ", "Prec", "Outcome"))
            print('{0} {1:>8} {2:>11} {3:>9} {4:>9} {5:>4} {6:>8} {7:>10} {8:>8} {9:>9} {10:>10} {11:>6} {12:>5} {13:>8}'.format("-"*len("Exon"), "-"*len("E_Frame") \
            , "-"*len("Exon_Start"), "-"*len("Exon_End"), "-"*len("E-Length"), "-"*len("Hit"), "-"*len("H_Frame"), "-"*len("Hit_Start"), "-"*len("Hit_End") \
            , "-"*len("H-Length"), "-"*len("H-Overlap"), "-"*len(" TPR "), "-"*len("Prec"), "-"*len("Outcome")))

            total_exon = 0
            total_hits = 0
            total_overlap = 0
            trues = set()
            for i in range(len(self.exons)):
                hits_indices = self.is_hit(self.exon_starts[i], self.exon_ends[i])

                trues = trues.union(set(hits_indices))
                exon_length = self.exon_ends[i] - self.exon_starts[i] + 1

                next = 0
                while True:
                    overlap = 0
                    if hits_indices:
                        overlap = min(self.exon_ends[i], max(self.amino_to_nucleic(self.hits_coordinates[hits_indices[next]][1] \
                            , self.hits_frame_number[hits_indices[next]]))) - max(self.exon_starts[i] \
                            , min(self.amino_to_nucleic(self.hits_coordinates[hits_indices[next]][0] \
                            , self.hits_frame_number[hits_indices[next]]))) + 1
                        total_overlap += overlap
                        hit_length = self.hits_coordinates[hits_indices[next]][1] - self.hits_coordinates[hits_indices[next]][0] + 1
                        total_hits += hit_length*3

                    exon_frame = self.exon_starts[i] % 3 if self.exon_starts[i] % 3 >= 1 else 3
                    hit_frame = -1 if not hits_indices else self.hits_frame_number[hits_indices[next]]

                    print('#' + '{0:>3} {1:>8} {2:>11} {3:>9} {4:>9} {5:>4} {6:>8} {7:>10} {8:>8} {9:>9} {10:>10} {11:>6.2f} {12:>5.2f} {13:>8}'.format(i+1 \
                        , exon_frame \
                        , self.exon_starts[i] \
                        , self.exon_ends[i] \
                        , exon_length
                        , 'NA ' if not hits_indices else hits_indices[next] + 1 \
                        , 'NA     ' if not hits_indices else self.hits_frame_number[hits_indices[next]] \
                        , 'NA       ' if not hits_indices else min(self.amino_to_nucleic(self.hits_coordinates[hits_indices[next]][0] \
                            , self.hits_frame_number[hits_indices[next]])) \
                        , 'NA     ' if not hits_indices else max(self.amino_to_nucleic(self.hits_coordinates[hits_indices[next]][1] \
                            , self.hits_frame_number[hits_indices[next]])) \
                        , 'NA      ' if not hits_indices else hit_length * 3 \
                        , 'NA       ' if not hits_indices else overlap \
                        , 0 if not hits_indices else overlap / exon_length \
                        , 0 if not hits_indices else overlap / (hit_length * 3) \
                        , 'FN' if not hits_indices else 'TP'))

                    if len(hits_indices) > next + 1:
                        next += 1
                    else:
                        break
                total_exon += exon_length

            self.FPs = set(range(len(self.hits_coordinates))).difference(trues)
            print()
            for i in self.FPs:
                hit_length = self.hits_coordinates[i][1] - self.hits_coordinates[i][0] + 1
                print('#' + '{0:>3} {1:>8} {2:>11} {3:>9} {4:>9} {5:>4} {6:>8} {7:>10} {8:>8} {9:>9} {10:>10} {11:>6.2f} {12:>5.2f} {13:>8}'.format('NA', 'NA', 'NA', 'NA' \
                    , 'NA', i+1, self.hits_frame_number[i], self.hits_coordinates[i][0], self.hits_coordinates[i][1] \
                    , self.hits_coordinates[i][1] - self.hits_coordinates[i][0] + 1, 'NA', 0, 0, 'FP'))
            print("\nAnnotation Information:\n", '-'*len("Annotation Information:"), sep='')
            print('{:<40} {:>5}'.format("Number of exons: ", len(self.exons)))
            print('{:<40} {:>5} {}'.format("Number of covered exons:", len(trues) , str("out  of  " + str(len(self.exons)) + ' = ' \
                + format(len(trues)/len(self.exons),'.2f'))))
            self.output = self.output + str(len(trues)) + '/' + str(len(self.exons)) + ', '
            print('{:<40} {:>5} {}'.format("Total exons size:", total_exon, "nucleotides"))
            print('{:<40} {:>5} {}'.format("Total overlap size:", total_overlap, "nucleotides"))
            print('{:<40} {:>5}'.format("Discovery score (O_TPR):", format(total_overlap/total_exon, '.2f')))
            print('{:<40} {:>5}'.format("Coding Percentage (O_PRC):", format(total_overlap/total_hits, '.2f')))
            self.output = self.output + format(total_overlap/total_exon, '.2f') + ', '
            self.output = self.output + format(total_overlap/total_hits, '.2f') + ', '
        if result != 'i':
            print()
            title = "Sequences of \"" + gene + "\" exons in the annotated file:"
            print(title)
            print('-'*len(title))
            for i in range(exon_index + 1):
                print('{0:>0} {1:>2} {2:>0} {3:>3}'.format("Exon #", str(i+1), " ", self.exons[i]))

        print()

    # Calculates codon frequencies of a sequence
    def codon_freq(self, sequence):
        sequence = sequence.upper()
        sequence = sequence.replace('T', 'U')

        start = end = 0 # codon coordinates

        codon_freq = [0] * len(self.genetic_code_vmit.keys())
        while (len(sequence) - end) / 3 >= 1:
            end = start + 3
            codon = sequence[start:end]

            if codon in self.genetic_code_vmit.keys():
                codon_freq[list(self.genetic_code_vmit.keys()).index(codon)] += 1
            start = end

        return codon_freq

    # Calculates codon frequencies of hits
    def codon_freq_hits(self):
        for index, hit_coord in enumerate(self.hits_coordinates):
            start = min(self.amino_to_nucleic(hit_coord[0], self.hits_frame_number[index]))
            end = max(self.amino_to_nucleic(hit_coord[1], self.hits_frame_number[index]))
            hit = self.dna[0][start-1:end]
            codon_freq = self.codon_freq(hit)
            self.hits_codon_frequencies.append(codon_freq)
        return self.codon_freq_norm([sum(i) for i in zip(*self.hits_codon_frequencies)])

    # Calculates codon frequencies of the query gene
    def codon_freq_gene(self):
        codon_freq = self.codon_freq(''.join(self.exons))
        return self.codon_freq_norm(codon_freq)

    # Calculates codon frequencies of genome
    def codon_freq_annot(self):
        # Reading the annotation string line by line
        lines = self.annotation.split('\n')

        # Finding the starting line of the FASTA file (>)
        current_line_index = -1
        current_line = re.sub(' +', ' ', lines[0].strip()).split(' ')
        while not current_line[0].startswith('>'):
            current_line_index = current_line_index + 1
            current_line = re.sub(' +', ' ', lines[current_line_index].strip()).split(' ')

        # Finding the starting line of the gene
        is_exon = False # it's an exon sequence line
        is_query = False
        temp_exon = '' # one exon holder
        temp_gene = [] # one gene holder
        index = 0

        # Looping until reaching the end of the genome
        while current_line_index < len(lines):
            current_line = re.sub(' +', ' ', lines[current_line_index]).strip().split(' ')
            # Filling buffer exon
            if is_exon and not current_line[0].startswith(';'):
                if current_line[0].isnumeric():
                    temp_exon += ''.join(current_line[1:])
                else:
                    temp_exon += ''.join(current_line)

            if current_line[0] == ';' and not len(re.findall('G-.*-', current_line[1])) and current_line[3] == 'start':
                if current_line[1][2:] == self.gene_name[0]:
                    is_query = True
                    self.gene_name[1] = index
                elif current_line[1][2:] in ['cob', 'cox1', 'cox2', 'nad1', 'nad5']:
                    is_query = False
            # exon starts
            if not is_query and current_line[0] == ';' and len(re.findall('G-.*-E\d*', current_line[1])) and current_line[3] == 'start':
                is_exon = True
            # exon ends
            elif not is_query and current_line[0] == ';' and len(re.findall('G-.*-E\d*', current_line[1])) and current_line[3] == 'end':
                is_exon = False
                temp_gene.append(temp_exon)
                temp_exon = ''
            # gene ends
            elif not is_query and current_line[0] == ';' and not len(re.findall('G-.*-', current_line[1])) and current_line[3] == 'end' and temp_gene != []:
                self.annot_genes.append(temp_gene)
                temp_gene = []
                index += 1
            current_line_index += 1

        codon_freq_exon = []
        for gene in self.annot_genes:
            for exon in gene:
                codon_freq_exon.append(self.codon_freq(exon))
            self.genome_codon_frequencies.append([sum(i) for i in zip(*codon_freq_exon)])


        return self.codon_freq_norm([sum(i) for i in zip(*self.genome_codon_frequencies)]), [sum(i) for i in zip(*self.genome_codon_frequencies)]

    # Normalizes a codon frequency vector
    def codon_freq_norm(self, codon_freq, precision = 3):
        if len(codon_freq) != 64:
            raise ValueError('Vector has not legnth 64.')

        amino_acids =  list(self.genetic_code_vmit.values())

        cdn_grp = {x : 0 for x in amino_acids}
        for i in range(64):
            cdn_grp[amino_acids[i]] += codon_freq[i]
        for k in cdn_grp.keys():
            if cdn_grp[k] == 0:
                cdn_grp[k] = 1
        cdn_grp = [cdn_grp[amino_acids[i]] for i in range(64)]
        return np.around(np.divide(codon_freq, cdn_grp), precision).tolist()

    # Calculates euclidean_distance, avg_dot_products and kl_divergence distances between two codon frequency distribution
    def codon_freq_dist(self, cdn_freq_ref, cdn_freq_qry):
        amino_acids =  list(self.genetic_code_vmit.values())

        cdn_grp_ref = {x : [] for x in amino_acids}
        for i in range(64):
            cdn_grp_ref[amino_acids[i]].append(cdn_freq_ref[i])

        cdn_grp_qry = {x : [] for x in amino_acids}
        for i in range(64):
            cdn_grp_qry[amino_acids[i]].append(cdn_freq_qry[i])

        # Removing the amino acids that are not present in the current hit
        for i in list(cdn_grp_qry):
            if not sum(cdn_grp_qry[i]):
                del cdn_grp_qry[i]

        new_cdn_freq_ref = np.asarray([cdn_freq_ref[i] for i in range(64) if amino_acids[i] in cdn_grp_qry.keys()])
        new_cdn_freq_qry = np.asarray([cdn_freq_qry[i] for i in range(64) if amino_acids[i] in cdn_grp_qry.keys()])

        for i in cdn_grp_qry.keys():
            cdn_grp_qry[i] = np.dot(cdn_grp_qry[i], cdn_grp_ref[i])

        euclidean_distance = np.linalg.norm(new_cdn_freq_ref - new_cdn_freq_qry)

        avg_dot_products = sum(cdn_grp_qry.values()) / len(cdn_grp_qry)

        kl_divergence = entropy(new_cdn_freq_qry, qk=new_cdn_freq_ref)

        return euclidean_distance, 1/avg_dot_products, kl_divergence, cdn_grp_qry

    # Prints codon frequency of a string in a genetic code table format
    def genetic_code_table(self, codon_freq, title=''):
        print('\n', title, '\n', '-'*len(title), '\n', sep='')

        n_a = ['U', 'C', 'A', 'G']
        print('{0:<8} {1:<12} {2:<12} {3:<12} {4:<12}'.format(" ", "U", "C", "A", "G"))
        print('{0:<8} {1:<12} {2:<12} {3:<12} {4:<12}'.format(" ", "-"*len("U"), "-"*len("C"), "-"*len("A"), "-"*len("G")))
        for i in range(16):
            print(n_a[i//4] + '{0:<3} {1:<12} {2:<12} {3:<12} {4:<12} {5:<12}'.format('' \
            , list(self.genetic_code_vmit.values())[i] + ' = ' + str(codon_freq[i]) \
            , list(self.genetic_code_vmit.values())[i+16] + ' = ' + str(codon_freq[i+16]) \
            , list(self.genetic_code_vmit.values())[i+32] + ' = ' + str(codon_freq[i+32]) \
            , list(self.genetic_code_vmit.values())[i+48] + ' = ' + str(codon_freq[i+48]) \
            , n_a[i - ((i//4)*4)]))
            if (i == 3 or i == 7 or i == 11):
                print('-'*58)

    # Reads HMMER output and extracts necessary information
    def hmmer_reader(self, hmmer_output):
        if hmmer_output == '':
            raise ValueError('HMMER output is invalid.')

        lines = hmmer_output.split('\n')
        for index, line in enumerate(lines):
            # Finding coordinates of hits in each sequence
            if line.startswith('>>'):
                read = index + 3
                # Reading the starting and the ending indices of each hmms, and hits
                while len(re.sub(' +', ' ', lines[read]).strip().split(' ')) == 16:
                    if re.sub(' +', ' ', line).strip().split(' ')[1][0].startswith('r'):
                        self.hits_frame_number.append(int(re.sub(' +', ' ', line).strip().split(' ')[1][1]))
                    else:
                        self.hits_frame_number.append(int(re.sub(' +', ' ', line).strip().split(' ')[1][0]))
                    self.hmms_coordinates.append([int(i) for i in re.sub(' +', ' ', lines[read]).strip().split(' ')[6:8]])
                    self.hits_coordinates.append([int(i) for i in re.sub(' +', ' ', lines[read]).strip().split(' ')[9:11]])
                    read += 1

        return self.hmms_coordinates, self.hits_coordinates

    def hmmsearch(self, hmm_profile):
        # Creating the translated_dna_file with 6 different sequences (ORFs) with 6 different recognizable titles
        print("\nForward ORFs:\n", '-'*len('Forward ORFs:'), sep='')
        translated_dna = ""
        for frame_number in range(1,4):
            print("Translating frame number ", str(frame_number), " ...")
            translated_frame = self.translate(self.dna[0], frame_number)
            translated_dna += self.string_to_fasta(translated_frame[0], title = (str(frame_number) + '_orf_' + self.dna[1]))

        print("\nReverse ORFs:\n", '-'*len('Reverse ORFs:'), sep='')
        for frame_number in range(1,4):
            print("Translating frame number ", str(frame_number), " ...")
            translated_frame = self.translate(self.dna[0], frame_number, True)
            translated_dna += self.string_to_fasta(translated_frame[0], title = ('r' + str(frame_number) + '_orf_' + self.dna[1]))

        # Writing the translated_frames_file to a file for HMMER use
        translated_frames_file = open("translated_frames.fasta", 'w+')
        translated_frames_file.write(translated_dna)
        seq_path = os.path.abspath("translated_frames.fasta")
        translated_frames_file.seek(0, 0)

        # "--nobias", "--incE", "10", "--incdomE", "10", "--F3", "1", "--nonull2", "--max"
        print("\nRunning HMMER ...\n")
        process = subprocess.run(["hmmsearch",  "--max", hmm_profile, seq_path], stdout=subprocess.PIPE, universal_newlines=True)
        hmmer_output = process.stdout
        return hmmer_output

    # TODO: This can be improved. Go next line only if overlap with prev hit.
    def hits_vis(self, hmms):
    	offset = 0
    	hit = ''
    	for i in range(len(hmms)):
    		hit = (hmms[i][1] - hmms[i][0] + 1)%self.window_width
    		offset = hmms[i][0]%self.window_width
    		if i and hmms[i][0] < hmms[i-1][1]:
    			print()
    		elif i:
    			offset = (hmms[i][0] - hmms[i-1][1])%self.window_width
    		print(' '*(offset-len(str(hmms[i-1][1]))), '' if hmms[i][0] == hmms[i-1][1] else hmms[i][0] \
    			, '-'*(hit - len(str(hmms[i][1])) - len(str(hmms[i][0]))), hmms[i][1], sep='', end='')
    	print()

    def cluster(self, codon_freqs_and_info):
        print("\nK-means Clustering Result:\n", '-'*len("K-means Clusterin Result:\n"), sep='')

        raw_codon_freqs = [item[1] for item in codon_freqs_and_info]
        amino_acids =  list(self.genetic_code_vmit.values())
        all_cdn_freq_norm = []
        all_aa_freqs=[]
        for codon_freq in raw_codon_freqs:
            aa_freqs = []
            cdn_freq_norm = self.codon_freq_norm(codon_freq)
            codon_groups = {x : [] for x in amino_acids}
            for i in range(64):
                codon_groups[amino_acids[i]].append(cdn_freq_norm[i])

            for i in codon_groups.keys():
                if not sum(codon_groups[i]):
                    aa_freqs.append(1)
                else:
                    aa_freqs.append(0)

            all_aa_freqs.append(aa_freqs)
            all_cdn_freq_norm.append(cdn_freq_norm)

        amino_freqs = sum(np.asarray(all_aa_freqs))

        plt.plot(amino_freqs.tolist())
        plt.title('Frequencies of Amino Acids')
        plt.xlabel('Amino Acids')
        plt.ylabel('Frequency')
        plt.xticks(list(range(len(amino_freqs))), codon_groups.keys(), rotation='vertical')
        #plt.show()

        included = [i for i, x in enumerate(amino_freqs.tolist()) if x >= 0.4 * len(self.hmms_coordinates)]
        aas = [list(codon_groups.keys())[i] for i in included]
        keys = [i for i , item in enumerate(amino_acids) if item in aas]
        data = [[item[i] for i in keys] for item in all_cdn_freq_norm]
        print_format = '{:>2} Cluster(s): '
        print_args = 0
        for i in range(2, len(self.hmms_coordinates)):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(data)
            print_data = [str(l) for l in sorted([sorted({j+1 for j , e in enumerate(kmeans.labels_) if e == item}) for item in range(0,i)])]
            while len(print_data) > print_args:
                print_format += '{:<' + str(len(print_data[print_args])+2) + '}'
                print_args += 1
            print(print_format.format(i, *print_data))

    # Represents a list in a nice format
    def list_rep(self, list):
        col = 5
        cur_line = 0
        print()
        for idx, item in enumerate(list):
            print('{:>5}:  {},     '.format(idx+1, item), end='')
            if idx//(col-1) > cur_line:
                print()
                cur_line += 1
        print()

    def cdn_bias_hm(self):
        genes = []
        count = 0
        for i in range(len(self.annot_genes) + 1):
            if i == self.gene_name[1]:
                genes.append(''.join(self.exons))
            else:
                genes.append(''.join(self.annot_genes[count]))
                count += 1
        freqs = []
        for i in self.annot_genes:
            print(''.join(i))
        print(self.gene_name[0], self.gene_name[1], ''.join(self.exons))
        genes = ''.join(genes)
        steps = list(range(0, len(''.join(genes))+1, int(3*(len(''.join(genes))/50))))
        for i in range(0, len(steps)-1):
            freqs.append(self.codon_freq(genes[steps[i]:steps[i+1]]))

        # sorted = []
        # s = [sum(i) for i in zip(*freqs)]
        # for i in freqs:
        #     print(s)
        #     print(i)
        #     sorted.append([x for _,x in sorted(zip(s,i))])

        fig, ax = plt.subplots()
        pos = ax.imshow(np.asarray(freqs), cmap='hot', interpolation='nearest', aspect='auto')
        fig.colorbar(pos)
        plt.xticks(list(range(len(self.genetic_code_vmit))), self.genetic_code_vmit.keys(), rotation='vertical')
        index = 1
        species = ''
        for line in self.annotation.split('\n'):
            line_elem = re.sub(' +', ' ', line.strip()).split(' ')
            if len(line_elem) >= 2 and line_elem[0].startswith('>'):
                species = re.sub('>', '', ''.join(line_elem[0:2]))
                break
        plt.savefig(species + '_' + self.gene_name[0] + '.png')
        plt.close()

    # Runs the code for a list of organisms and genes
    def run_all(self, data_file):
        try:
            os.remove('output.txt')
        except:
            pass

        data = open(data_file, 'r').read().split('\n')
        for run in data:
            # if not run.startswith('#') and not run.startswith('*'):
            if not run.startswith('#'):
                parameters = re.sub(' +', ' ', run.strip()).split(' ')
                print('Parameters: ', parameters)
                t = Translated_Search()
                t.run(*parameters)
        print()

    def run(self, dna_seq, hmm_profile, annotation, gene, format):
        print()
        self.window_width = os.get_terminal_size()[0]
        # Reading the DNA sequence and HMM protein profile as arguments
        print("Reading the DNA file ...")
        self.dna = self.fasta_to_string(open(dna_seq, 'r').read())

        # Running HMMER on DNA sequence
        hmmer_output = self.hmmsearch(hmm_profile)
        print(hmmer_output)

        print('-'*self.window_width)
        print(' '*(int(self.window_width/2)-8), 'Exon Hunter (EH)')
        print('-'*self.window_width)

        # Getting HMM and Hit coordinates
        hmms, hits = self.hmmer_reader(hmmer_output)
        self.output = '/' + str(len(hmms))
        if not len(hmms):
            raise ValueError('No HMMER hit found.')

        print("\nHMMER hits (sorted): ", len(hmms), '\n', '-'*(len('HMMER hits (sorted): ')+len(str(len(hmms)))), sep='')
        hmms_sorted = sorted(hmms)
        self.list_rep(hmms_sorted)
        opt_hmms = self.combinations(list(map(tuple, hmms)), set(), len(hmms))
        opt_hmms_sorted = sorted([list(i) for i in opt_hmms[1]])
        self.output = str(len(opt_hmms_sorted)) + self.output + ', '
        print("\nOptimal hit set score: ", opt_hmms[0])
        print("Number of included hits: ", len(opt_hmms_sorted))
        # TODO: 'Crossing detection report' should be implemented in a filtering way
        print(self.message)
        print("Removed hit(s): ", [("Index: "+str(self.hmms_coordinates.index(list(i))+1)+", Coordinates: "+ str(i)) for i in set(map(tuple, hmms)).difference(set(map(tuple, opt_hmms_sorted)))])
        self.output = self.output + '[' + '-'.join([(str(self.hmms_coordinates.index(list(i))+1)) for i in set(map(tuple, hmms)).difference(set(map(tuple, opt_hmms_sorted)))]) + ']' + ', '
        print("\nOptimal hit set:\n", '-'*len("Optimal hit set:"), sep='')
        self.list_rep(opt_hmms_sorted)

        print("\n\nHits positioning visualization:\n", '-'*len("Hits positioning visualization:"), sep='')
        print("All:\n")
        self.hits_vis(hmms_sorted)
        print("\nOptimal:\n")
        self.hits_vis(opt_hmms_sorted)

        self.annotation = open(annotation, 'r').read()
        self.annot_valid()

        self.gene_name[0] = gene
        self.exons_finder()
        self.annot_info(format)
        cdn_freq_genome = self.codon_freq_annot()
        self.genetic_code_table(cdn_freq_genome[0], "Codon Frequency of Genome:")
        # Printing the count of each amino acids
        codon_counts = {x : [] for x in list(self.genetic_code_vmit.values())}
        for i in range(64):
            codon_counts[list(self.genetic_code_vmit.values())[i]].append(cdn_freq_genome[1][i])
        print()
        for i in codon_counts:
            codon_counts[i] = sum(codon_counts[i])
            print(i, ': ', codon_counts[i], ', ', end='')
        print()

        cdn_freq_gene = self.codon_freq_gene()
        self.genetic_code_table(cdn_freq_gene, str("Codon Frequency of Gene '" + self.gene_name[0] + "':"))
        cdn_freq_hits = self.codon_freq_hits()
        self.genetic_code_table(cdn_freq_hits, "Avg. Codon Frequency of hits:")

        print()
        self.cdn_bias_hm()
        print()

        eu, dot, kl, grp = self.codon_freq_dist(cdn_freq_genome[0], cdn_freq_gene)
        codon_freqs_and_info = [0]*len(hmms)

        for i, hit_codon_freq in enumerate(self.hits_codon_frequencies):
            length = hmms[i][1] - hmms[i][0] + 1
            euclidean_distance, avg_dot_products, kl_divergence, cdn_grps = self.codon_freq_dist(cdn_freq_genome[0], self.codon_freq_norm(hit_codon_freq))
            codon_freqs_and_info[i] = [i, hit_codon_freq, length, euclidean_distance, euclidean_distance / length, kl_divergence, kl_divergence / length, avg_dot_products, cdn_grps]

        ax = plt.subplot(1,1,1)
        for i in [3, 5, 7]:
            my_color = ''
            if i == 3:
                my_color = 'green'
            elif i == 5:
                my_color = 'red'
            else:
                my_color = 'blue'
            # ax.plot(np.asarray([item[i] for item in codon_freqs_and_info])/max([e for e in [item[i] for item in codon_freqs_and_info] if math.isfinite(e)] + [eu, kl, dot][(i//2)-1]), color = my_color)
            # plt.axhline(y=[eu, kl][(i//2)-1]/max([e for e in [item[i] for item in codon_freqs_and_info] if math.isfinite(e)] + [eu, dot, kl][(i//2)-1]), color = my_color)
            ax.plot(np.asarray([item[i] for item in codon_freqs_and_info]), color = my_color)
            plt.axhline(y=[eu, kl, dot][(i//2)-1], color = my_color)
        for i in self.FPs:
            plt.axvline(x=i, color = 'black')

        plt.title('Distance distribution of Hits (Euc = G and KL_D = R, Dot_P = B)')
        plt.xlabel('Hits')
        plt.ylabel('Distance')
        #plt.show()

        print("\n\nCodon Frequency of Hits:\n", '-'*len("Codon Frequency of Hits:"), sep='')
        sorting_measure = 5
        measures = {0 : "Index", 1 : "Codon_Freq", 2 : "Length", 3 : "Euclidean_Distance", 4 : "Weighted_Euclidean_Distance", 5 : "KL_Divergence", 6 : "Weighted_KL_Divergence", 7 : "Average_Dot_Product"}
        print("Distance ranking of each Hit based on ", measures[sorting_measure], ' (Increasing): ', [item[0]+1 for item in sorted(codon_freqs_and_info, key=lambda tup: tup[sorting_measure], reverse = False)])
        self.output = self.output + str([item[0]+1 for item in sorted(codon_freqs_and_info, key=lambda tup: tup[sorting_measure], reverse = False)])

        for hit_info in sorted(codon_freqs_and_info, key=lambda tup: tup[sorting_measure], reverse = False):
            self.genetic_code_table(self.codon_freq_norm(hit_info[1]), ("Rank " + str(i) + ", Hit #" + str(hit_info[0]+1)))
            print("\n\tLength: \n\t", hit_info[2], sep='')
            print("\n\tEuclidean, Weighted Euclidean: \n\t%.5f, %.5f" % (hit_info[3], hit_info[4]))
            print("\n\tKL-divergence, Weighted KL-divergence: \n\t%.5f, %.5f" % (hit_info[5], hit_info[6]))
            print("\n\tKL-divergence (reverse): \n\t", entropy(cdn_freq_genome[0], qk=self.codon_freq_norm(hit_info[1])) / length, sep='')
            # print("\n\tdot product: \n\t%.5f" % (hit_info[7]))
            print("\n\tAvg. of dot products of each AA: \n\t%s" % str(hit_info[7]))
            print("\n\tAAs dot product: \n\t%s" % str(hit_info[8]))
            print()

        # self.cluster(codon_freqs_and_info)

        print('\n')
        print(self.output)
        out = open ('output.txt', 'a')
        out.write(self.output + '\n')


if sys.argv[1] == '-h':
    print("python3 self.py dna_sequence = "" hmm_profile = "", annotation = "", gene = "", output_format = """)
elif len(sys.argv) == 2:
    t = Translated_Search()
    t.run_all(sys.argv[1])
elif len(sys.argv) == 6:
    t = Translated_Search()
    t.run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
else:
    raise ValueError('Invalid number of arguments.')

#python3 self.py /Users/ahaji060/Documents/Thesis/forMarcel/83introns_annot_Endoconidiophora_resinifera_strain_1410B.fasta-stripped /Users/ahaji060/Documents/Thesis/forMarcel/cob_protein_profile.hmm /Users/ahaji060/Documents/Thesis/forMarcel/83introns_annot_Endoconidiophora_resinifera_strain_1410B.fasta 'cob' i
