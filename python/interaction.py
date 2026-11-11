import numpy as np
import scipy.stats
import copy


class SingleParticleEnergy:
    """
    Class representing a single particle energy (SPE).
    
    Attributes:
        value (float): The current value (or mean/default value if sampling).
        unc (float): Uncertainty used for sampling. If positive, samples are generated as `value * [-unc, unc]`.
        minval (float): Minimum value for sampling (used if `unc <= 0`).
        maxval (float): Maximum value for sampling (used if `unc <= 0`).
        key (tuple): A unique key identifying the SPE.
    """

    def __init__(self, index, value, *args):
        """
        Initialize a SingleParticleEnergy object.

        Args:
            index (int): Index of the SPE.
            value (float): Default value of the SPE.
            *args: Optional arguments to define uncertainty or min/max values.
                   - If 1 argument: unc
                   - If 2 arguments: minval, maxval
                   - If 3 arguments: unc, minval, maxval
        """
        self.unc = -1.0
        self.minval = 0.0
        self.maxval = 0.0
        self.value = value
        self.key = (index, index, -1, -1, -1, -1)

        if args:
            if len(args) == 1:
                self.unc = args[0]
            elif len(args) == 2:
                self.minval, self.maxval = args
            elif len(args) == 3:
                self.unc, self.minval, self.maxval = args

    def __str__(self):
        """String representation of the SPE."""
        return f"{self.value:2.5f}"

    def set_value(self, value):
        """
        Set the value of the SPE.

        Args:
            value (float): New value to set.
        """
        self.value = float(value)


class TwoBodyMatrixElement:
    """
    Class representing a two-body matrix element (TBME).
    
    Attributes:
        i1, i2, i3, i4 (int): Indices of the two-body interaction.
        J (int): Total angular momentum.
        T (int): Isospin quantum number.
        value (float): Value of the TBME.
        key (tuple): A unique key identifying the TBME.
    """

    def __init__(self, i, j, k, l, J, T, value, *args):
        """
        Initialize a TwoBodyMatrixElement object.

        Args:
            i, j, k, l (int): Indices of the two-body interaction.
            J (int): Total angular momentum.
            T (int): Isospin quantum number.
            value (float): Default value of the TBME.
            *args: Optional arguments to define uncertainty or min/max values.
                   - If 1 argument: unc
                   - If 2 arguments: minval, maxval
                   - If 3 arguments: unc, minval, maxval
        """
        self.i1 = i
        self.i2 = j
        self.i3 = k
        self.i4 = l
        self.J = J
        self.T = T
        self.value = value
        self.key = (i, j, k, l, J, T)

        if args:
            if len(args) == 1:
                self.unc = args[0]
            elif len(args) == 2:
                self.minval, self.maxval = args
            elif len(args) == 3:
                self.unc, self.minval, self.maxval = args

    def __str__(self):
        """String representation of the TBME."""
        return f" {self.i1} {self.i2} {self.i3} {self.i4} {self.J} {self.T} {self.value:2.5f}"

    def set_value(self, value):
        """
        Set the value of the TBME.

        Args:
            value (float): New value to set.
        """
        self.value = float(value)


class Interaction:
    """
    Class representing an interaction file containing SPEs and TBMEs.
    
    Attributes:
        filename (str): Name of the interaction file.
        linecount (int): Number of lines in the file.
        spes (list): List of SingleParticleEnergy objects.
        tbmes (list): List of TwoBodyMatrixElement objects.
        print_spes (list): List of SPEs for printing.
        print_tbmes (list): List of TBMEs for printing.
        particleNumberScaled (bool): Whether the particle number is scaled.
        A, Ap2, power (float): Parameters for particle number scaling.
    """

    def __init__(self, filename):
        """
        Initialize an Interaction object by parsing an interaction file.

        Args:
            filename (str): Path to the interaction file.
        """
        self.filename = filename
        self.linecount = 0
        self.spes = []
        self.tbmes = []
        self.print_spes = []
        self.print_tbmes = []
        self.particleNumberScaled = False
        self.A = 0
        self.Ap2 = 0
        self.power = 0.0

        is_spe_line = True
        with open(filename, 'r') as file:
            for line in file:
                if line.startswith(('#', '!')):
                    continue

                words = line.split()
                if is_spe_line:
                    self.linecount = int(words[0])
                    if self.linecount < 0:
                        self.particleNumberScaled = True
                        self.linecount = abs(self.linecount)

                    offset = 3 if self.particleNumberScaled else 0
                    for i in range(1, len(words) - offset):
                        self.spes.append(SingleParticleEnergy(i, float(words[i]), 0.1))

                    if self.particleNumberScaled:
                        self.A, self.Ap2, self.power = map(float, words[-3:])
                    is_spe_line = False
                    continue

                i1, i2, i3, i4, J, T = map(int, words[:6])
                value = float(words[6])
                self.tbmes.append(TwoBodyMatrixElement(i1, i2, i3, i4, J, T, value, 0.1))

        self.print_spes = copy.deepcopy(self.spes)
        self.print_tbmes = copy.deepcopy(self.tbmes)

    def __str__(self):
        """String representation of the interaction."""
        lc = -self.linecount if self.particleNumberScaled else self.linecount
        result = f"{lc} "
        result += " ".join(str(spe) for spe in self.print_spes)
        if self.particleNumberScaled:
            result += f" {self.A} {self.Ap2} {self.power}"
        result += "\n"
        result += "\n".join(str(tbme) for tbme in self.print_tbmes)
        return result

    def get_matrix_elements(self):
        """
        Get the matrix elements of the interaction as a numpy array.

        Returns:
            np.ndarray: Array of matrix elements.
        """
        coefficients = np.zeros(self.size())
        interaction_dict = dict(sorted(self.get_matrix_elements_as_dict().items()))
        for i, (_, value) in enumerate(interaction_dict.items()):
            coefficients[i] = value
        return coefficients

    def set_matrix_elements(self, coefficients):
        """
        Set the matrix elements of the interaction using a numpy array.

        Args:
            coefficients (np.ndarray): Array of matrix elements.
        """
        interaction_dict = dict(sorted(self.get_matrix_elements_as_dict().items()))
        for i, key in enumerate(interaction_dict.keys()):
            interaction_dict[key] = coefficients[i]
        self.set_matrix_element_values_by_key(interaction_dict)

    def get_matrix_elements_as_dict(self):
        """
        Get the matrix elements as a dictionary.

        Returns:
            dict: Dictionary of matrix elements with keys as identifiers.
        """
        elements = {spe.key: spe.value for spe in self.print_spes}
        elements.update({tbme.key: tbme.value for tbme in self.print_tbmes})
        return dict(sorted(elements.items()))

    def set_matrix_element_values_by_key(self, element_dict):
        """
        Set the values of the matrix elements using a dictionary.

        Args:
            element_dict (dict): Dictionary of matrix elements with keys as identifiers.
        """
        for key, value in element_dict.items():
            found = False
            for spe in self.print_spes:
                if key == spe.key:
                    spe.value = value
                    found = True
                    break
            for tbme in self.print_tbmes:
                if key == tbme.key:
                    tbme.value = value
                    found = True
                    break
            if not found:
                print(f"Did not find matrix element in interaction: {key}")

    def size(self):
        """
        Get the total number of SPEs and TBMEs.

        Returns:
            int: Total number of SPEs and TBMEs.
        """
        return len(self.spes) + len(self.tbmes)

    def reset_values(self):
        """
        Reset the values of SPEs and TBMEs to their initial state.
        """
        self.print_spes = copy.deepcopy(self.spes)
        self.print_tbmes = copy.deepcopy(self.tbmes)

    def orderby(self, reference_interaction, verbose=False):
        """
        Reorder the TBME indices to match the order of a reference interaction.

        This method ensures that the current interaction's matrix elements
        are aligned with the ordering of the reference interaction. If a key
        is not found, it attempts to find valid aliases.

        Args:
            reference_interaction (Interaction): The reference interaction to match.
            verbose (bool): If True, prints debug information during reordering.
        """
        current_dict = self.get_matrix_elements_as_dict()
        reference_dict = reference_interaction.get_matrix_elements_as_dict()
        reordered_dict = {}

        # Define the angular momentum values for the single-particle states
        speJ = [3.0 / 2, 5.0 / 2, 1.0 / 2]

        for reference_key, _ in reference_dict.items():
            if reference_key in current_dict:
                if verbose:
                    print(f"Found reference key {reference_key} in current keys.")
                value = current_dict[reference_key]
            else:
                if verbose:
                    print(f"Searching for aliases of {reference_key}...")
                aliases = self.make_aliases(reference_key, speJ)
                for alias, phase in aliases.items():
                    if alias in current_dict:
                        value = phase * current_dict[alias]
                        if verbose:
                            print(f"Found alias: {alias} (phase = {phase})")
                        break
                else:
                    print(f"Could not find {reference_key} in the current interaction!")
                    continue
            reordered_dict[reference_key] = value

        self.initbyDict(reordered_dict)

    def make_aliases(self, key, speJ):
        """
        Create a dictionary of valid reorderings (aliases) for a given key.

        This method computes the phase factor required to correct the matrix
        element when reordering the indices of a two-body matrix element.

        Args:
            key (tuple): The key representing a matrix element (a, b, c, d, J, T).
            speJ (list): List of angular momentum values for single-particle states.

        Returns:
            dict: A dictionary where keys are reordered tuples, and values are
                  the corresponding phase factors.
        """
        a, b, c, d, J, T = key
        ja = speJ[a - 1]
        jb = speJ[b - 1]
        jc = speJ[c - 1]
        jd = speJ[d - 1]

        # SPE case: no aliases
        if c == -1:
            return {(a, b, c, d, J, T): 0}

        # TBME case: 8 possible aliases
        aliases = {
            (a, b, c, d, J, T): 1,
            (b, a, c, d, J, T): -(-1) ** (ja + jb + 1 - J - T),
            (a, b, d, c, J, T): -(-1) ** (jc + jd + 1 - J - T),
            (b, a, d, c, J, T): (-1) ** (ja + jb + jc + jd),
            (c, d, a, b, J, T): 1,
            (c, d, b, a, J, T): -(-1) ** (ja + jb + 1 - J - T),
            (d, c, a, b, J, T): -(-1) ** (jc + jd + 1 - J - T),
            (d, c, b, a, J, T): (-1) ** (ja + jb + jc + jd),
        }
        return aliases

    def initbyDict(self, new_dict):
        """
        Reset all matrix elements and initialize them from a dictionary.

        Args:
            new_dict (dict): Dictionary of matrix elements with keys as identifiers.
        """
        self.spes = []
        self.tbmes = []
        self.print_spes = []
        self.print_tbmes = []

        for key, value in new_dict.items():
            a, b, c, d, J, T = key
            if c == -1:  # SPE case
                self.spes.append(SingleParticleEnergy(a, value))
            else:  # TBME case
                self.tbmes.append(TwoBodyMatrixElement(a, b, c, d, J, T, value))

        self.reset_values()
