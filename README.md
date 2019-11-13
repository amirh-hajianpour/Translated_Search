# Translated Search

Genome annotation using translated DNA and HMMER. Given a FASTA DNA file and a Hidden Markov Model (HMM) Protein Profile file, this application locates the protein in the given DNA.

## To run:

python3 Translated_Search.py [-h:help] 
                             <FASTA: DNA file>
                             <hmm: protein_profile file>
                             <MasterFile: TRUE annotation file>
                             <string: gene_name>
                             <output format: i|s|b>

i: indeices, s: sequences, b: both

## Example:

```
python3 Translated_Search.py some_dna.fasta
                             some_protein_profile.hmm
                             some_dna_annotation.mf
                             "some_gene"
                             'i'
```
