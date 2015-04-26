# -------------------------------------
# Outside imports
# -------------------------------------
import numpy as np
import itertools as it
from collections import OrderedDict

# -------------------------------------
# Local imports
# -------------------------------------
from epistasis.core.utils import hamming_distance, find_differences, enumerate_space

# -------------------------------------
# Main class for building epistasis map
# -------------------------------------

class BaseMap(object):
    
    def _map(self, keys, values):
        """ Return ordered dictionary mapping two properties in self. """
        return OrderedDict([(keys[i], values[i]) for i in range(self._n)])
        
    def _if_dict(self, dictionary):
        """ If setter method is passed a dictionary with genotypes as keys, 
            use those keys to populate array of elements in order
        """
        elements = np.empty(self._n, dtype=float)
        for i in range(self._n):
            elements[i] = dictionary[self._genotypes[i]]
        return elements
    

class BinaryMap(BaseMap):
    
    @property
    def bits(self):
        """ Get Binary representation of genotypes. """
        return self._bits
        
    @property
    def bit_indices(self):
        """ Get indices of genotypes in self.genotypes that mapped to their binary representation. """
        return self._bit_indices
        
    @property
    def bit_phenotypes(self):
        """ Get the phenotype values in an array orderd same as binary reprentation. """
        return self.phenotypes[self.bit_indices]
        
    @property
    def bit_phenotype_errors(self):
        """ Get the phenotype values in an array orderd same as binary reprentation. """
        if self.log_transform is True:
            return np.array((self.phenotype_errors[0,self.bit_indices],self.phenotype_errors[1,self.bit_indices]))
        else:
            return self.phenotype_errors[self.bit_indices]
            
    @property
    def bit2pheno(self):
        """ Return dict of genotypes mapped to phenotypes. """
        return self._map(self.bits, self.phenotypes[self.bit_indices])
        
    @property
    def geno2binary(self):
        """ Return dictionary of genotypes mapped to their binary representation. """
        mapping = dict()
        for i in range(self.n):
            mapping[self.genotypes[self.bit_indices[i]]] = self.bits[i] 
        return mapping
    

class InteractionMap(BaseMap):
    
    @property
    def log_transform(self):
        """ Boolean argument telling whether space is log transformed. """
        return self._log_transform
        
    @property
    def interaction_values(self):
        """ Get the values of the interaction in the system"""
        return self._interaction_values
        
    @property
    def interaction_errors(self):
        """ Get the value of the interaction errors in the system. """
        return self._interaction_errors


    @property
    def interaction_labels(self):
        """ Get the interaction labels, which describe the position of interacting mutations in
            the genotypes. (type==list of lists, see self._build_interaction_labels)
        """
        return self._interaction_labels
        
    @property
    def interaction_keys(self):
        """ Get the interaction keys. (type==list of str, see self._build_interaction_labels)"""
        return self._interaction_keys
        
    @property
    def interaction_indices(self):
        """ Get the interaction index in interaction matrix. """
        return self._interaction_indices
        
    @property
    def interaction_genotypes(self):
        """ Get the interaction genotype. """
        elements = ['w.t.']
        for label in self._interaction_labels[1:]:
            elements.append(self._label_to_genotype(label))
        return elements
        
    @property
    def key2value(self):
        """ Return dict of interaction keys mapped to their values. """
        return OrderedDict([(self.interaction_keys[i], self.interaction_values[i]) for i in range(len(self.interaction_values))])
        
    @property
    def genotype2value(self):
        """ Return dict of interaction genotypes mapped to their values. """
        return OrderedDict([(self.interaction_genotypes[i], self.interaction_values[i]) for i in range(len(self.interaction_values))])
        return self._map(self.interaction_genotypes, self.interaction_values)
        
    @property
    def genotype2error(self):
        """ Return dict of interaction genotypes mapped to their values. """
        return OrderedDict([(self.interaction_genotypes[i], self.interaction_errors[:,i]) for i in range(len(self.interaction_values))])

    # ----------------------------------------------
    # Setter Functions
    # ----------------------------------------------
        
    @interaction_values.setter
    def interaction_values(self, interaction_values):
        """ Set the interactions of the system, set by an Epistasis model (see ..models.py)."""
        if len(interaction_values) != len(self._interaction_labels):
            raise Exception("Number of interactions give to map is different than was defined. ")
        self._interaction_values = interaction_values
        
    @interaction_errors.setter
    def interaction_errors(self, interaction_errors):
        """ Set the interaction errors of the system, set by an Epistasis model (see ..models.py)."""
        if self.log_transform is True:
            if np.array(interaction_errors).shape != (2, len(self._interaction_labels)):
                raise Exception("""interaction_errors is not the right shape (should include 2 elements
                                    for each interaction, upper and lower bounds).""")
        else:
            if len(interaction_errors) != len(self._interaction_labels):    
                raise Exception("Number of interactions give to map is different than was defined. ")
        self._interaction_errors = interaction_errors


    @log_transform.setter
    def log_transform(self, boolean):
        """ True/False to log transform the space. """
        self._log_transform = boolean
        
    # ----------------------------------------------
    # Methods
    # ----------------------------------------------    

    def _build_interaction_map(self):
        """ Returns a label and key for every epistatic interaction. 
            
            Also returns a dictionary with order mapped to the index in the interactions array.
            
            An interaction label looks like [1,4,6] (type==list).
            An interaction key looks like '1,4,6'   (type==str).
        """
        labels = [[0]]
        keys = ['0']
        order_indices = dict()
        for o in range(1,self.order+1):
            start = len(labels)
            for label in it.combinations(range(1,self.n_mutations+1), o):
                labels.append(list(label))
                key = ','.join([str(i) for i in label])
                keys.append(key)
            stop = len(labels)
            order_indices[o] = [start, stop]
        return labels, keys, order_indices


    def _label_to_genotype(self, label):
        """ Convert a label (list(3,4,5)) to its genotype representation ('A3V, A4V, A5V'). """
        genotype = ""
        for l in label:
            # Labels are offset by 1, remove offset for wildtype/mutation array index
            array_index = l - 1
            mutation = self.wildtype[array_index] + str(l) + self.mutations[array_index]
            genotype += mutation + ','
        # Return genotype without the last comma
        return genotype[:-1]

class EpistasisMap(BaseMap):
    
    
    def __init__(self):
    """
        Object that maps epistasis in a genotype-phenotype map. 
        
        Attributes:gi to
        ----------
        length: int, 
            length of genotypes
        n: int
            size of genotype-phenotype map
        order: int
            order of epistasis in system
        wildtype: str
            wildtype genotype
        mutations: array of chars
            individual mutations from wildtype that are in the system
        genotypes: array
            genotype in system
        phenotypes: array
            quantitative phenotypes in system
        phenotype_errors: array
            errors for each phenotype value
        indices: array
            genotype indices
        interaction_values: array
            epistatic interactions in the genotype-phenotype map
        interaction_error: array
            errors for each epistatic interaction
        interaction_indices: array
            indices for interaction's position in mutation matrix
        interaction_genotypes: array
            interactions as their mutation labels
        interaction_labels: list of lists
            List of interactions indices 
        interaction_keys: list
            List of interaction keys
    """
        self.Interactions = InteractionMap()
        self.Binary = BinaryMap()
        
    # ------------------------------------------------------
    # Getter methods for attributes that can be set by user.
    # ------------------------------------------------------
    
    @property
    def length(self):
        """ Get length of the genotypes. """
        return self._length    
    
    @property
    def n(self):
        """ Get number of genotypes, i.e. size of the system. """
        return self._n

    @property
    def order(self):
        """ Get order of epistasis in system. """
        return self._order
    
    @property
    def log_transform(self):
        """ Boolean argument telling whether space is log transformed. """
        return self._log_transform
        
    @property
    def wildtype(self):
        """ Get reference genotypes for interactions. """
        return self._wildtype
        
    @property
    def mutant(self):
        """ Get the furthest genotype from the wildtype genotype. """
        return self._mutant

    @property
    def genotypes(self):
        """ Get the genotypes of the system. """
        return self._genotypes
        
    @property
    def phenotypes(self):
        """ Get the phenotypes of the system. """
        return self._phenotypes
    
    @property
    def phenotype_errors(self):
        """ Get the phenotypes' errors in the system. """
        return self._phenotype_errors
        
                
    # ------------------------------------------------------------------
    # Getter methods for attributes that are not set explicitly by user.
    # ------------------------------------------------------------------
    @property
    def indices(self):
        """ Return numpy array of genotypes position. """
        return self._indices
        
    @property
    def mutations(self):
        """ Get possible that occur from reference system. """
        return self._mutations
        
    @property
    def mutation_indices(self):
        """ Get the indices of mutations in the sequence. """
        return self._mutation_indices
        
    @property
    def n_mutations(self):
        """ Get the number of mutations in the space. """
        return self._n_mutations
        
    # ----------------------------------------------------------
    # Getter methods for mapping objects
    # ----------------------------------------------------------   
    
    @property
    def geno2pheno(self):
        """ Return dict of genotypes mapped to phenotypes. """
        return self._map(self.genotypes, self.phenotypes)

    @property
    def geno2index(self):
        """ Return dict of genotypes mapped to their indices in transition matrix. """
        return self._map(self.genotypes, self.indices)
        
    # ----------------------------------------------------------
    # Setter methods
    # ----------------------------------------------------------
    
    @log_transform.setter
    def log_transform(self, boolean):
        """ True/False to log transform the space. """
        self._log_transform = boolean
    
    @genotypes.setter
    def genotypes(self, genotypes):
        """ Set genotypes from ordered list of sequences. """
        self._n = len(genotypes)
        self._length = len(genotypes[0])
        self._genotypes = np.array(genotypes)
        self._indices = np.arange(self._n)
        
    @wildtype.setter
    def wildtype(self, wildtype):
        """ Set the reference genotype among the mutants in the system. """
        self._wildtype = wildtype
        self._mutant = self._farthest_genotype(wildtype)
        self._mutation_indices = find_differences(self.wildtype, self.mutant)
        self._mutations = [self.mutant[i] for i in self.mutation_indices]
        self._n_mutations = len(self._mutations)
        self._to_bits()
    
    @order.setter
    def order(self, order):
        """ Set the order of epistasis in the system. As a consequence, 
            this mapping object creates the """
        self._order = order
        # Create interaction labels and keys
        self._interaction_labels, self._interaction_keys, self._order_indices = self._build_interaction_map()
        self._interaction_indices = np.arange(len(self._interaction_labels))
        
    @phenotypes.setter
    def phenotypes(self, phenotypes):
        """ NORMALIZE and set phenotypes from ordered list of phenotypes 
            
            Args:
            -----
            phenotypes: array-like or dict
                if array-like, it musted be ordered by genotype; if dict,
                this method automatically orders the phenotypes into numpy
                array.
        """
        if type(phenotypes) is dict:
            self._phenotypes = self._if_dict(phenotypes)/phenotypes[self.wildtype]
        else:
            if len(phenotypes) != len(self._genotypes):
                raise("Number of phenotypes does not equal number of genotypes.")
            else:
                wildtype_index = self.geno2index[self.wildtype]
                self._phenotypes = phenotypes/phenotypes[wildtype_index] 

        # log transform if log_transform = True
        if self.log_transform is True:
            self._phenotypes = np.log10(self._phenotypes)

        
    @phenotype_errors.setter
    def phenotype_errors(self, errors):
        """ Set error from ordered list of phenotype error. 
            
            Args:
            -----
            error: array-like or dict
                if array-like, it musted be ordered by genotype; if dict,
                this method automatically orders the errors into numpy
                array.
        """
        # Order phenotype errors from geno2pheno_err dictionary
        if type(errors) is dict:
            errors = self._if_dict(phenotype_errors)
        
        # For log-transformations of error, need to translate errors to center around 1,
        # then take the log.
        if self.log_transform is True:
            errors = np.array((np.log10(1-errors), np.log10(1 + errors)))
        
        self._phenotype_errors = errors
    
    # ---------------------------------
    # Useful methods for mapping object
    # ---------------------------------

    def _differ_all_sites(self, reference):
        """ Find the genotype in the system that differs at all sites from reference.""" 
        for genotype in self.genotypes:
            if hamming_distance(genotype, reference) == self._length:
                differs = genotype
                break
        return differs

    def _farthest_genotype(self, reference):
        """ Find the genotype in the system that differs at the most sites. """ 
        mutations = 0
        for genotype in self.genotypes:
            differs = hamming_distance(genotype, reference)
            if differs > mutations:
                mutations = int(differs)
                mutant = str(genotype)
        return mutant

    def _to_bits(self):
        """ Encode the genotypes an ordered binary set of genotypes with 
            wildtype as reference state (ref is all zeros).
            
            Essentially, this method maps each genotype to their binary representation
            relative to the 'wildtype' sequence.
        """
        w = list(self.wildtype)
        m = list(self.mutations)

        # get genotype indices
        geno2index = self.geno2index
        
        # Build binary space
        # this is a really slow/memory intensive step ... need to revisit this
        full_genotypes, binaries = enumerate_space(self.wildtype, self.mutant, binary = True)
        bin2geno = dict(zip(binaries, full_genotypes))
        bits = list()
        bit_indices = list()
        # initialize bit_indicies
        for b in binaries:
            try:
                bit_indices.append(geno2index[bin2geno[b]])
                bits.append(b)
            except:
                pass
        self._bits = np.array(bits)
        self._bit_indices = np.array(bit_indices)        
    