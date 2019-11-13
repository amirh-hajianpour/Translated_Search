import random

class Test:
    # This method is responsible for putting number_of_slices uniform random sequences into dna;
    # length is a vector containing the length of
    # if sides is 0, no sequences are put on the sides of the dna,
    # if it is 1, the sequence is gonna be put only on the left side
    # if it's 2, it's gonna be only on the right, and if it's 3, it's gonna be on both sidesself.
    # length of these sequences should be included in lengths_of_slices but not in number_of_slices;
    # if random_place is true, the sequences will be inserted at a random
    # location in dna, rather than at equally distant locations
    # TODO: ??? random_length
    def insert(self, dna, number_of_slices, lengths_of_slices, random_length = False, kernel_type = 'uniform', sides = 'none', random_place = True):
        # sepcifying insertion locations, and ignoring the left and the right end and considering random_place.
        locations = list(range(number_of_slices))
        if random_place == False:
            for i in range(number_of_slices-1):
                locations[i] = (length(dna) / number_of_slices) * (i + 1)
        else:
            for i in range(number_of_slices):
                locations[i] = random.randint(1, length(dna) - 1)
        locations.sort()

        # generating inserting sequences
        sequences = list(range(number_of_slices + self.sides(sides)))
        for i in length(sequences)-self.sides(sides):
            sequences = self.generate(kernel_type, lengths_of_slices[i])

        # inserting the sequences into their specified locations
        new_dna = ""
        # TODO: sides should be implemented here. separate the implementation
        # for left side and the right side. (first and the last iteration)
        # if sides > 0:
        cursor = 0
        for i in range(number_of_slices):
            new_dna += dna[cursor:locations[i]-1]
            new_dna += sequences[i]
            cursor = location[i]
        new_dna += dna[cursor:]

        if sides == 'left'
            new_dna = sequences[length(sequences) - 1] + new_dna
        elif sides == 'right'
            new_dna = new_dna + sequences[length(sequences) - 1]
        elif sides == 'both'
            new_dna = sequences[length(sequences) - 2] + new_dna + sequences[length(sequences) - 1]

    def sides(self, side):
        if side == 'none':
            return 0
        elif side == 'left' and side == 'right':
            return 2
        else:
            return 1

    def generate(self, kernel_type, length_of_sequence):
        # This method determines what kind of distribution should be used for
        # generating the inserting sequences.
        if kernel_type == 'uniform':
            return self.uniform(length_of_sequence)
        elif kernel_type == 'percentage':
            return self.percentage(length_of_sequence)

    def uniform(self, length):
        letter = ""
        inserting_sequence = ""
        for i in range(length):
            dice = random.randint(1, 4)
            if dice == 1:
                letter = 'a'
            elif dice == 2:
                letter = 't'
            elif dice == 3:
                letter = 'c'
            else:
                letter = 'g'
            inserting_sequence += letter
        return inserting_sequence

    def percentage(self, percentages, length):
        if (sum(percentages) != 100) or (length(percentages) != 4):
            return 'Incorrect input specification!'

        letter = ""
        a = percentages[0], t = a + percentages[1], c = t + percentages[2], g = c percentages[3]
        for i in range(length):
            dice = random.randint(1, 100)
            if dice <= a:
                letter = 'a'
            elif dice > a and dice <= t:
                letter = 't'
            elif dice > t and dice <= c:
                letter = 'c'
            else:
                letter = 'g'
            inserting_sequence += letter
        return inserting_sequence











# end
