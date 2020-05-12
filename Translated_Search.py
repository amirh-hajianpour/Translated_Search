import sys
import subprocess
import os
import re
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
from scipy.stats import entropy
from scipy.spatial import distance_matrix
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

class Translated_Search:
    MAX_EXONS = 100
    dna = ""
    exons = [''] * MAX_EXONS
    genes = [''] * MAX_EXONS
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
        'AUG': 'Met', 'AUA': 'Met',
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
        'UGG': 'Trp', 'UGA': 'Trp',
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
    hits_codon_frequencies = []
    annot_codon_frequencies = []
    hmms_coordinates = []
    hits_coordinates = []

    def __init__(self):
        print()

    @staticmethod
    # Takes a position in amino acid coordinates (translated DNA) to nucleic acid coordinates (DNA)
    # Returns a nucleic acid position
    def amino_to_nucleic(aa_pos, frame_number = 0):
        start = aa_pos * 3 + frame_number
        end = aa_pos * 3 + 2 + frame_number

        return start, end

    @staticmethod
    # Takes a position in nucleic acid coordinates (DNA) to amino acid coordinates (translated DNA)
    # Returns an amino acid position
    def nucleic_to_amino(na_pos, frame_number = 0):
        return (na_pos - frame_number) // 3

    @staticmethod
    # Detects whether a sequence is DNA, RNA, Protein, or invalid
    # Returns the type of the sequence
    def sequence_type(sequence):
        sequence = sequence.upper()

        letters =  set()
        for letter in sequence:
            letters.add(letter)

        if letters.issubset({'A', 'T', 'C', 'G'}):
            return 'DNA'
        elif letters.issubset({'A', 'U', 'C', 'G'}):
            return 'RNA'
        elif letters.issubset({'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'}):
            return 'Protein'

        print("Invalid Sequence!")

    @staticmethod
    # Makes a complementary DNA strand
    # Returns complementary DNA string
    def complementor(dna):
        dna = dna.upper()
        c_dna = dna

        # T => A, and A => T
        c_dna = c_dna.replace('A', '1')
        c_dna = c_dna.replace('T', 'A')
        c_dna = c_dna.replace('1', 'T')

        # G => C, and C => G
        c_dna = c_dna.replace('C', '1')
        c_dna = c_dna.replace('G', 'C')
        c_dna = c_dna.replace('1', 'G')

        return c_dna

    @staticmethod
    # Transcribes and Translates a DNA string
    # Returns the translated DNA, list of abnormal codons, and whether the length of the DNA was divisible by 3 or not
    def translator(my_dna, frame_number = 0, reverse = False, stop_codon = '*'):
        my_dna = my_dna.upper()
        transcribed_dna = my_dna[frame_number:]

        # Changing STOP codon translating symbol
        Translated_Search.abbreviation.update(STOP = str(stop_codon))

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
            # If reverse ORF, codons should be read backward
            if reverse:
                codon = codon[::-1]
            # Abnormal codons are skipped. (e.g. AAUPUU => AAU = N)
            if str(codon) in Translated_Search.genetic_code_vmit.keys():
                amino_acid = Translated_Search.genetic_code_vmit[str(codon)]
                translated_dna += Translated_Search.abbreviation[amino_acid]
            else:
                abnormal_codons.append(str(codon))
            start = end

        return translated_dna, abnormal_codons, True if (len(transcribed_dna) % 3 == 0) else False

    @staticmethod
    # Converts a string to FASTA format
    # Returns a FASTA Format string with specified length and title
    def string_to_fasta(sequence, title = "", line_length = 80):
        fasta = ""
        start = end = 0
        # Getting each line of the FASTA Format
        while len(sequence) - end >= line_length:
            end = start + line_length
            fasta += sequence[start:end] + '\n'
            start = end

        # Adding last line
        if len(sequence) - end > 0:
            fasta += sequence[start:len(sequence)] + '\n'

        # Adding title as first line
        if title != "":
            fasta = '>' + title + '\n' + fasta + '\n'
        else:
            fasta = '>' + 'Unknown' + '\n' + fasta + '\n'
        return fasta

    @staticmethod
    # Converts a FASTA format to string
    # Returns the title and the sequence
    def fasta_to_string(fasta):
        sequence = ""
        for line in fasta.split('\n'):
            # Ignoring the comments
            if not line.startswith(';'):
                # Getting the title line
                if line.startswith('>'):
                    title = line.replace('>', '')
                # Getting the whole sequence as a one-line string
                else:
                    sequence += line

        return sequence, title

    @staticmethod
    # Computes a score for a set of hits considering:
    # (1) overlap as penalty, and (2) match (coverage) as positive score
    # Rejects sets of hits that have crossings
    def scoring_function(hits_set):
        hits_set = list(map(list, hits_set))
        # If the set only contains one hit
        if len(hits_set) == 1:
            return hits_set[0][1] - hits_set[0][0] + 1

        # Checking for crossing (invalid order of) hits
        hits_set = sorted(hits_set, key=lambda tup: tup[0])
        indices = [-1]*len(hits_set)
        i = 0
        for hit in hits_set:
            indices[i] = Translated_Search.hmms_coordinates.index(hit)
            i += 1
        hits = [Translated_Search.hits_coordinates[i] for i in indices]
        if hits != sorted(hits, key=lambda tup: tup[0]):
            return -1000000

        # Looping through all pairs of hits to calculate the overall overlap
        overlap = 0
        for i in range(len(hits_set)):
            for j in range(i + 1, len(hits_set)):
                if max(hits_set[i][0], hits_set[j][0]) < min(hits_set[i][1], hits_set[j][1]):
                    overlap += min(hits_set[i][1], hits_set[j][1]) - max(hits_set[i][0], hits_set[j][0]) + 1

        # Calculating the coverage (ovrelap is being added 2 times)
        coverage = 0
        for i in range(len(hits_set)):
            coverage += hits_set[i][1] - hits_set[i][0] + 1

        return coverage - (2*overlap)

    # TODO: A new and more efficient algorithm should be implemented for this method
    @staticmethod
    # Creates a binary tree in which a hit is selected (left) or not (right): 2^hits
    def naive(hits, bag_of_hits, n):
        # Base Case
        if n == 0:
            return Translated_Search.scoring_function(bag_of_hits), bag_of_hits

        # Keeping the bag unchanged
        old_bag_of_hits = bag_of_hits.copy()
        # Adding the last element of the hits list to the bag
        bag_of_hits.add(hits[n - 1])
        # Calculating the score of the bag if n-1th hit was added
        left_score, left_set = Translated_Search.naive(hits, bag_of_hits, n - 1)
        # Calculating the score of the bag if n-1th hit was not added
        right_score, right_set = Translated_Search.naive(hits, old_bag_of_hits, n - 1)

        # Keeping the item if it led to a better score
        if left_score >= right_score:
            return left_score, left_set
        # Dropping the item if it didn't lead to a better score
        else:
            return right_score, right_set

    @staticmethod
    # Checks if an exon is found in the search and returns a boolean in addition to the overlap
    def is_hit(exon_start, exon_end):
        num_of_overlaps = 0
        hits_indices = [-1]*len(Translated_Search.hits_coordinates)
        for i in range(len(Translated_Search.hits_coordinates)):
            hit = Translated_Search.hits_coordinates[i].copy()
            # Going from Amino Acid coordinates to Nucleic Acid coordinates
            hit[0] = min(Translated_Search.amino_to_nucleic(hit[0]))
            hit[1] = max(Translated_Search.amino_to_nucleic(hit[1]))
            # if they have overlap
            if max(exon_start, hit[0]) < min(exon_end, hit[1]):
                hits_indices[num_of_overlaps] = i
                num_of_overlaps += 1
        overlap_hits_indices = [0]*num_of_overlaps
        for i in range(num_of_overlaps):
            if(hits_indices[i] != -1):
                overlap_hits_indices[i] = hits_indices[i]

        return num_of_overlaps != 0, overlap_hits_indices

    # TODO: Separate exon extractor and hit print methods
    # TODO: is_hit() might return multiple hits. It should be considered.
    @staticmethod
    # Extracts exons of a gene in a master file format (result: indices ('i') and sequences('s'))
    def mf_exon_reader(mf_file, gene, result = 'i'):
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
        exon_starts = [0] * Translated_Search.MAX_EXONS # starting indices of exons
        exon_ends = [0] * Translated_Search.MAX_EXONS   # ending indices of exons
        exon = [""] * Translated_Search.MAX_EXONS   # list of sequences of exons
        temp_exon = [""] * Translated_Search.MAX_EXONS  # buffer exon
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
                for i in range(current_line_index - exon_start_line):
                    exon[exon_number] = exon[exon_number] + temp_exon[i]
                temp_exon = [""] * Translated_Search.MAX_EXONS
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
            print()
            title = "Positions of \"" + gene + "\" exons in the annotated file:"
            print(title)
            print('-'*len(title))
            print('{0} {1:>11} {2:>9} {3:>9} {4:>10} {5:>8} {6:>9} {7:>10} {8:>8}'.format("Exon", "Exon_Start", "Exon_End", "E-Length", "Hit_Start", "Hit_End" \
                , "H-Length", "H-Overlap", "H-Score"))
            print('{0} {1:>11} {2:>9} {3:>9} {4:>10} {5:>8} {6:>9} {7:>10} {8:>8}'.format("-"*len("Exon"), "-"*len("Exon_Start"), "-"*len("Exon_End"), "-"*len("E-Length") \
                , "-"*len("Hit_Start"), "-"*len("Hit_End"), "-"*len("H-Length"), "-"*len("H-Overlap"), "-"*len("H-Score")))

            total_exon_length = 0
            total_overlap_length = 0
            j = 0
            overlap = 0
            for i in range(exon_number + 1):
                hits_indices = Translated_Search.is_hit(exon_starts[i], exon_ends[i])[1]
                if (len(hits_indices) == 0):
                    hit_start = -1
                    hit_end = -1
                else:
                    hit_start = Translated_Search.hits_coordinates[hits_indices[0]][0]
                    hit_end = Translated_Search.hits_coordinates[hits_indices[0]][1]
                total_exon_length += exon_ends[i] - exon_starts[i] + 1
                overlap = min(exon_ends[i], max(Translated_Search.amino_to_nucleic(hit_end))) - \
                    max(exon_starts[i], min(Translated_Search.amino_to_nucleic(hit_start))) + 1
                if(overlap > 0):
                    total_overlap_length += overlap
                else:
                    overlap = 0
                print('#' + '{0:>3} {1:>11} {2:>9} {3:>9} {4:>10} {5:>8} {6:>9} {7:>10} {8:>8.2f}'.format(i+1, exon_starts[i], exon_ends[i] \
                    , exon_ends[i] - exon_starts[i] + 1, hit_start, hit_end, hit_end - hit_start + 1, overlap, overlap / (exon_ends[i] - exon_starts[i] + 1)))
                for k in range(len(hits_indices)-1):
                    print('{0:>5} {1:>10}'.format(Translated_Search.hits_coordinates[hits_indices[k]][0]))
            print("\nOverall annotation statistics summary:")
            print("--------------------------------------")
            print('{0:<40} {1:>3} {2:<13}'.format("Total exon size:", total_exon_length, "nucleotides"))
            print('{0:<40} {1:>3} {2:<13}'.format("Total overlap length:", total_overlap_length, "nucleotides"))
            print('{0:<40} {1:>4} {2:}'.format("Number of found exons (TPR):", hits_indices[0] + 1, str("out of " + str(exon_number + 1) + ' = ' + format(len(Translated_Search.hits_coordinates)/(exon_number + 1),'.2f'))))
            print('{0:<40} {1:>3}'.format("Coverage score:", format(total_overlap_length/total_exon_length, '.2f')))
        if result != 'i':
            print()
            title = "Sequences of \"" + gene + "\" exons in the annotated file:"
            print(title)
            print('-'*len(title))
            for i in range(exon_number + 1):
                print('{0:>0} {1:>2} {2:>0} {3:>3}'.format("Exon #", str(i+1), " ", exon[i]))
        Translated_Search.exons = exon[0:exon_number+1]
        print()
        return exon_starts, exon_ends

    @staticmethod
    # Calculates codon frequencies of a string
    def codon_freq(sequence):
        sequence = sequence.upper()

        sequence = sequence.replace('T', 'U')

        start = end = 0 # codon coordinates

        codon_freq = [0] * len(Translated_Search.genetic_code_vmit.keys())
        while (len(sequence) - end) / 3 >= 1:
            end = start + 3
            codon = sequence[start:end]

            if str(codon) in Translated_Search.genetic_code_vmit.keys():
                codon_freq[list(Translated_Search.genetic_code_vmit.keys()).index(str(codon))]+=1
            start = end
        return codon_freq

    @staticmethod
    # Calculates codon frequencies of hits
    def codon_freq_hits():
        hit_number = 0
        Translated_Search.hits_codon_frequencies = [0] * len(Translated_Search.hits_coordinates)
        for hit_coord in Translated_Search.hits_coordinates:
            hit = Translated_Search.dna[0][hit_coord[0]:hit_coord[1]+1]
            codon_freq = Translated_Search.codon_freq(hit)
            Translated_Search.hits_codon_frequencies[hit_number] = codon_freq
            hit_number += 1
        return Translated_Search.hits_codon_frequencies

    @staticmethod
    # Calculates codon frequencies of a gene
    def codon_freq_annot():
        hit_number = 0
        Translated_Search.annot_codon_frequencies = [0] * len(Translated_Search.exons)
        for exon in Translated_Search.exons:
            codon_freq = Translated_Search.codon_freq(exon)
            Translated_Search.annot_codon_frequencies[hit_number] = codon_freq
            hit_number += 1
        return Translated_Search.annot_codon_frequencies

    @staticmethod
    # Calculates codon frequencies of all genes
    def codon_freq_annot_2(mf_file):
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
        # while current_line[0] != ';' or not re.findall('G-.*-E.*', current_line[1]) or current_line[3] != 'start':
        #     current_line_index = current_line_index + 1
        #     current_line = re.sub(' +', ' ', lines[current_line_index]).split(' ')

        is_exon = False # it's an exon sequence line
        exon_number = -1    # current number of the exon
        exon = [""] * 1000   # list of sequences of exons
        temp_exon = [""] * Translated_Search.MAX_EXONS  # buffer exon
        exon_start_line = 0 # starting line of the exon

        # Looping until reaching end of the gene
        while current_line_index < len(lines) - 1:
            # exon starts
            if current_line[0] == ';' and len(re.findall('G-.*-E\d*', current_line[1])) != 0 and current_line[3] == 'start':
                is_exon = True
                exon_number = exon_number + 1
                # Looking for the start index of the current exon
                temp_line_index = current_line_index
                temp_line = current_line
                while temp_line[0].startswith(';'):
                    temp_line_index = temp_line_index + 1
                    temp_line = re.sub(' +', ' ', lines[temp_line_index]).strip().split(' ')
                exon_start_line = temp_line_index
            # exon ends
            elif current_line[0] == ';' and len(re.findall('G-.*-E\d*', current_line[1])) != 0 and current_line[3] == 'end':
                is_exon = False
                # Copying buffer exon to list of sequences of exons
                for i in range(current_line_index - exon_start_line):
                    exon[exon_number] = exon[exon_number] + temp_exon[i]
                temp_exon = [""] * Translated_Search.MAX_EXONS
                # Looking for the end index of the current exon
                temp_line_index = current_line_index
                temp_line = current_line
                while temp_line[0].startswith(';'):
                    temp_line_index = temp_line_index + 1
                    temp_line = re.sub(' +', ' ', lines[temp_line_index]).strip().split(' ')
            current_line_index = current_line_index + 1
            current_line = re.sub(' +', ' ', lines[current_line_index]).strip().split(' ')
            # Filling buffer exon
            # Note: this does not allow other inner struucture in exon
            if is_exon and not current_line[0].startswith(';'):
                temp_exon[current_line_index - exon_start_line] = temp_exon[current_line_index - exon_start_line] + current_line[1]

        Translated_Search.genes = exon[0:exon_number+1]
        hit_number = 0
        Translated_Search.annot_codon_frequencies = [0] * len(Translated_Search.genes)
        for exon in Translated_Search.genes:
            codon_freq = Translated_Search.codon_freq(exon)
            Translated_Search.annot_codon_frequencies[hit_number] = codon_freq
            hit_number += 1
        return Translated_Search.annot_codon_frequencies

    # Normalizes a codon frequency vector
    def codon_freq_norm(codon_freq, precision = 3):

        amino_acid =  list(Translated_Search.genetic_code_vmit.values())

        if (len(codon_freq) == 64):
            sums = {x : 0 for x in amino_acid}
            for i in range(64):
                sums[amino_acid[i]] += codon_freq[i]
            for k in sums.keys():
                if sums[k] == 0:
                    sums[k] = 1
            sums = [sums[amino_acid[i]] for i in range(64)]
            return np.around(np.divide(codon_freq, sums), precision)
        else:
            print("Vector has not a legnth of 64.")

    def genetic_code_table(codon_freq, title=''):
        print("\n", title, sep='')
        print('-'*len(title), '\n', sep='')

        n_a = ['U', 'C', 'A', 'G']
        print('{0:<8} {1:<12} {2:<12} {3:<12} {4:<12}'.format(" ", "U", "C", "A", "G"))
        print('{0:<8} {1:<12} {2:<12} {3:<12} {4:<12}'.format(" ", "-"*len("U"), "-"*len("C"), "-"*len("A"), "-"*len("G")))
        for i in range(16):
            print(n_a[i//4] + '{0:<3} {1:<12} {2:<12} {3:<12} {4:<12} {5:<12}'.format('' \
            , list(Translated_Search.genetic_code_vmit.values())[i] + ' = ' + str(codon_freq[i]) \
            , list(Translated_Search.genetic_code_vmit.values())[i+16] + ' = ' + str(codon_freq[i+16]) \
            , list(Translated_Search.genetic_code_vmit.values())[i+32] + ' = ' + str(codon_freq[i+32]) \
            , list(Translated_Search.genetic_code_vmit.values())[i+48] + ' = ' + str(codon_freq[i+48]) \
            , n_a[i - ((i//4)*4)]))
            if (i == 3 or i == 7 or i == 11):
                print('-'*58)

    @staticmethod
    # Reads HMMER output and extracts necessary information
    def hmmer_reader(hmmer_output):
        line_counter = 0
        current_hit = 0
        lines = hmmer_output.split('\n')
        for line in lines:
            current_line = re.sub(' +', ' ', line).strip().split(' ')
            # Finding the number of hits
            if current_line[0] == 'Scores' and current_line[1] == 'for' and current_line[2] == 'complete' and current_line[3] == 'sequences':
                read = line_counter + 4
                if len(re.sub(' +', ' ', lines[read]).strip().split(' ')) == 0:
                    return "No hit found!", "No hit found!"
                number_of_hits = 0
                while len(re.sub(' +', ' ', lines[read]).strip()) != 0:
                    if len(re.sub(' +', ' ', lines[read]).strip().split(' ')) >= 9:
                        number_of_hits += int(re.sub(' +', ' ', lines[read]).strip().split(' ')[7])
                    read += 1
                Translated_Search.hmms_coordinates = list(range(number_of_hits))
                Translated_Search.hits_coordinates = list(range(number_of_hits))
            # Finding hits in each sequence
            if current_line[0] == '>>':
                read = line_counter + 3
                # Reading the start and the end index of each hit
                while len(re.sub(' +', ' ', lines[read]).strip().split(' ')) == 16:
                    Translated_Search.hmms_coordinates[current_hit] = list(map(int, re.sub(' +', ' ', lines[read]).strip().split(' ')[6:8]))
                    Translated_Search.hits_coordinates[current_hit] = list(map(int, re.sub(' +', ' ', lines[read]).strip().split(' ')[9:11]))
                    read += 1
                    current_hit += 1
            line_counter += 1

        return Translated_Search.hmms_coordinates, Translated_Search.hits_coordinates

    @staticmethod
    def hmmsearch(hmm_profile, dna_seq):
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
            print("Translating frame number ", str(frame_number + 1), " ...")
            translated_frame = Translated_Search.translator(Translated_Search.dna[0], frame_number)
            translated_dna = translated_dna + Translated_Search.string_to_fasta(translated_frame[0], title = (str(frame_number+1) + '_orf_' + Translated_Search.dna[1]))

        print("\nReverse ORFs:")
        print("-------------")
        for frame_number in range(3):
            print("Translating frame number ", str(frame_number + 1), " ...")
            translated_frame = Translated_Search.translator(Translated_Search.complementor(Translated_Search.dna[0]), frame_number, True)
            translated_dna = translated_dna + Translated_Search.string_to_fasta(translated_frame[0], title = ('r' + str(frame_number+1) + '_orf_' + Translated_Search.dna[1]))

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
        return Translated_Search.hmmer_reader(output)

    @staticmethod
    def run(dna_seq, hmm_profile, annotated, gene, format):
        print()
        hmms, hits = Translated_Search.hmmsearch(hmm_profile, dna_seq)
        ex = Translated_Search.mf_exon_reader(annotated, gene, format)
        return hmms, hits


if sys.argv[1] == '-h':
    print("python3 Translation.py dna_seq = "", hmm_profile = "", annotated = "", gene = "", format = """)
elif (len(sys.argv) == 3):
    hmms, hits = Translated_Search.hmmsearch(sys.argv[1], sys.argv[2])
elif (len(sys.argv) == 6):
    hmms, hits = Translated_Search.run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])


if hmms != "No hit found!":
    print("All HMMER hits (sorted): ", len(hmms))
    hmmer_hits = sorted(list(map(tuple, hmms)), key=lambda tup: tup[0])
    print(hmmer_hits)
    my_out = Translated_Search.naive(list(map(tuple, hmms)), set(), len(hmms))
else:
    print(hmms)

optimal_hits = sorted(list(map(tuple, my_out[1])), key=lambda tup: tup[0])
window_width = os.get_terminal_size()[0]

print("\nHighest score is: ", my_out[0])
print("Number of included hits: ", len(optimal_hits))
print("The best set of hits is: \n", optimal_hits, sep='')
text = "Hits positioning visualization:"
print('\n', text, '\n', '-'*len(text), sep='')
print("All:\n")
for i in range(len(hmms)):
    print(' '*((hmmer_hits[i][0]-1)%window_width), hmmer_hits[i][0], \
        '-'*((hmmer_hits[i][1] - hmmer_hits[i][0] - len(str(hmmer_hits[i][1])) - len(str(hmmer_hits[i][0])) + 1)%window_width), hmmer_hits[i][1], sep='')
print('\n')
print("Optimal:\n")
for i in range(len(my_out[1])):
    print(' '*((optimal_hits[i][0]-1)%window_width), optimal_hits[i][0], \
        '-'*((optimal_hits[i][1] - optimal_hits[i][0] - len(str(optimal_hits[i][1])) - len(str(optimal_hits[i][0])) + 1)%window_width), optimal_hits[i][1], sep='')
print('\n')

codon_freq_annot = Translated_Search.codon_freq_annot_2(sys.argv[3])
codon_freq_annot = [sum(i) for i in zip(*codon_freq_annot)]

codon_freq_annot_norm = Translated_Search.codon_freq_norm(codon_freq_annot)
Translated_Search.genetic_code_table(codon_freq_annot_norm, "Codon Frequency of Annotation:")

codon_freqs_and_info = [0]*len(hmms)

i = -1
amino_acid = list(Translated_Search.genetic_code_vmit.values())
hit_codon_groups = {x : [] for x in amino_acid}
for j in range(64):
    hit_codon_groups[amino_acid[j]].append(codon_freq_annot_norm[j])

for hit_codon_freq in Translated_Search.codon_freq_hits():
    i += 1
    hit_codon_freq_norm = Translated_Search.codon_freq_norm(hit_codon_freq)
    length = hmms[i][1] - hmms[i][0] + 1
    euclidea_distance = np.linalg.norm(hit_codon_freq_norm - codon_freq_annot_norm)
    kl_divergence = entropy(hit_codon_freq_norm, qk=codon_freq_annot_norm)
    #dot_product = np.dot(hit_codon_freq_norm, codon_freq_annot_norm)

    groups_dot_products = {x : [] for x in amino_acid}
    for j in range(64):
        groups_dot_products[amino_acid[j]].append(hit_codon_freq_norm[j])

    for k in hit_codon_groups.keys():
        if sum(groups_dot_products[k]) == 0:
            del groups_dot_products[k]

    for k in groups_dot_products.keys():
        groups_dot_products[k] = np.dot(groups_dot_products[k], hit_codon_groups[k])

    sum_group_dot_products = sum(groups_dot_products.values()) / len(groups_dot_products)


    codon_freqs_and_info[i] = [i, hit_codon_freq, length, euclidea_distance, euclidea_distance / length, kl_divergence, kl_divergence / length, sum_group_dot_products, groups_dot_products]

print("\n\nCodon Frequency of Hits:")
print('-'*len("Codon Frequency of Hits:"), '\n', sep='')
for hit_info in sorted(codon_freqs_and_info, key=lambda tup: tup[7], reverse = False):
    Translated_Search.genetic_code_table(Translated_Search.codon_freq_norm(hit_info[1]), ("Hit #" + str(hit_info[0]+1)))
    print("\n\tLength: \n\t", hit_info[2], sep='')
    print("\n\tEuclidean, Weighted Euclidean: \n\t%.5f:, %.5f" % (hit_info[3], hit_info[4]))
    print("\n\tKL-divergence, Weighted KL-divergence: \n\t%.5f, %.5f" % (hit_info[5], hit_info[6]))
    print("\n\tKL-divergence (reverse): \n\t", entropy(codon_freq_annot_norm, qk=Translated_Search.codon_freq_norm(hit_info[1])) / length, sep='')
    #print("\n\tdot product: \n\t%.5f" % (hit_info[7]))
    print("\n\tAvg. of dot products of each AA: \n\t%s" % str(hit_info[7]))
    print("\n\tAAs dot product: \n\t%s" % str(hit_info[8]))
    print()

nodes = [0]* (len(hmms) + 1)
nodes[0] = codon_freq_annot_norm
for i in range(1, len(hmms) + 1):
    nodes[i] = Translated_Search.codon_freq_norm(codon_freqs_and_info[i-1][1])

df = pd.DataFrame(nodes, columns=list(Translated_Search.genetic_code_vmit.values()), index=list(range(len(hmms)+1)))
dm = np.array(pd.DataFrame(distance_matrix(df.values, df.values), index=df.index, columns=df.index)).tolist()
G = nx.Graph()
for i in range(len(hmms)+1):
    for j in range(len(hmms)+1):
        if i != j:
            G.add_edges_from([(i,j)], weight=1/np.array(dm).tolist()[i][j])

pos = nx.spring_layout(G)
nx.draw(G, with_labels = True)
plt.show()

print("\nK-means Clustering Result:")
print('-'*len("K-means Clusterin Result:"), '\n', sep='')
wcss = []
fmt = ' '.join('Hit #{:<4}' for r in range(len(hmms)))
print(' '*7, fmt.format(*list(range(1,len(hmms)+1))), 'Number of Clusters')

for i in range(len(hmms)):
    kmeans = KMeans(n_clusters=i+1, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit([item[1] for item in codon_freqs_and_info])
    wcss.append(kmeans.inertia_)
    fmt = ' '.join('{:>9}' for r in range(len(hmms)))
    print(' '*2, fmt.format(*kmeans.labels_+1), ' '*12, i+1)
plt.plot(range(len(hmms)), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

print('\n')


#python3 Translated_Search.py /Users/ahaji060/Documents/Thesis/forMarcel/83introns_annot_Endoconidiophora_resinifera_strain_1410B.fasta-stripped /Users/ahaji060/Documents/Thesis/forMarcel/cob_protein_profile.hmm /Users/ahaji060/Documents/Thesis/forMarcel/83introns_annot_Endoconidiophora_resinifera_strain_1410B.fasta 'cob' i
