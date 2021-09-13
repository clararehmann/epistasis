
import traceback
import warnings
import sys

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback
warnings.simplefilter("always")

import pytest

@pytest.fixture(scope="module")
def test_data():
    """
    Set of maps with different characteristics to test.
    """

    test_data = []

    # Complete binary dataset
    test_data.append({"wildtype":"AAA",
                      "genotype":["AAA", "AAB", "ABA", "ABB",
                                  "BAA", "BAB", "BBA", "BBB"],
                      "all_possible_genotype":["AAA", "AAB", "ABA", "ABB",
                                                "BAA", "BAB", "BBA", "BBB"],
                      "phenotype":[0.2611278, 0.60470609, 0.13114308, 0.76428437,
                                    0.5018751, 0.18654072, 0.88086482, 0.18263346],
                      "uncertainty":[0.05, 0.05, 0.05, 0.05,
                                       0.05, 0.05, 0.05, 0.05],
                      "binary":["000", "001", "010", "011",
                                "100", "101", "110", "111"],
                      "all_possible_binary":["000", "001", "010", "011",
                                "100", "101", "110", "111"],
                      "name":["wildtype","A2B",   "A1B",    "A1B/A2B",
                              "A0B",     "A0B/A2B","A0B/A1B","A0B/A1B/A2B"],
                      "all_possible_name":["wildtype","A2B",   "A1B",    "A1B/A2B",
                                            "A0B",     "A0B/A2B","A0B/A1B","A0B/A1B/A2B"],
                      "n_mutations":[0,1,1,2,1,2,2,3],
                      "mutations":[["A", "B"],
                                   ["A", "B"],
                                   ["A", "B"]],
                      "length":3,
                      "n":8,
                      "mutant":"BBB"})


    # Binary at two sites, trinary at third.
    test_data.append({"wildtype":"AAA",
                      "genotype":["AAA", "AAB", "AAC", "ABA",
                                  "ABB", "ABC", "BAA", "BAB",
                                  "BAC", "BBA", "BBB", "BBC"],
                      "all_possible_genotype":["AAA", "AAB", "AAC", "ABA",
                                                "ABB", "ABC", "BAA", "BAB",
                                                "BAC", "BBA", "BBB", "BBC"],
                      "phenotype":[0.60371285, 0.10893567, 0.49704416, 0.34674266,
                                    0.26102007, 0.02631915, 0.44587924, 0.31596652,
                                    0.87037953, 0.95649285, 0.39668621, 0.66987709],
                      "uncertainty":[0.05, 0.05, 0.05, 0.05,
                                       0.05, 0.05, 0.05, 0.05,
                                       0.05, 0.05, 0.05, 0.05],
                      "binary":['0000', '0010', '0001', '0100',
                                '0110', '0101', '1000', '1010',
                                '1001', '1100', '1110', '1101'],
                      "all_possible_binary":['0000', '0010', '0001', '0100',
                                             '0110', '0101', '1000', '1010',
                                             '1001', '1100', '1110', '1101'],
                      "name":["wildtype","A2B",    "A2C",        "A1B",
                              "A1B/A2B", "A1B/A2C","A0B",        "A0B/A2B",
                              "A0B/A2C", "A0B/A1B","A0B/A1B/A2B","A0B/A1B/A2C"],
                      "all_possible_name":["wildtype","A2B",    "A2C",        "A2B",
                                            "A1B/A2B", "A1B/A2C","A0B",        "A0B/A2B",
                                            "A0B/A2C", "A0B/A1B","A0B/A1B/A2B","A0B/A1B/A2C"],
                      "n_mutations":[0,1,1,1,2,2,1,2,2,2,3,3],
                      "mutations":[["A", "B"],
                                   ["A", "B"],
                                   ["A", "B","C"]],
                      "length":3,
                      "n":12,
                      "mutant":"BBC"})

    # Binary, complete with invariant position
    test_data.append({"wildtype":"AAA",
                      "genotype":["AAA", "AAB",
                                  "BAA", "BAB"],
                      "all_possible_genotype":["AAA", "AAB", "BAA","BAB"],
                      "phenotype":[0.2611278, 0.60470609,
                                    0.5018751, 0.18654072],
                      "uncertainty":[0.05, 0.05,
                                       0.05, 0.05],
                      "binary":["00", "01","10", "11"],
                      "all_possible_binary":["00", "01","10", "11"],
                      "name":["wildtype","A2B","A0B","A0B/A2B"],
                      "all_possible_name":["wildtype","A2B","A0B","A0B/A2B"],
                      "n_mutations":[0,1,1,2],
                      "mutations":[["A", "B"],
                                   ["A"],
                                   ["A", "B"]],
                      "length":3,
                      "n":4,
                      "mutant":"BAB"})

    # Binary, incomplete
    test_data.append({"wildtype":"AA",
                      "genotype":["AA", "AB","BA"],
                      "all_possible_genotype":["AA", "AB", "BA","BB"],
                      "phenotype":[0.2611278, 0.60470609,0.5018751],
                      "uncertainty":[0.05, 0.05,0.05],
                      "binary":["00", "01","10"],
                      "all_possible_binary":["00", "01","10","11"],
                      "name":["wildtype","A1B","A0B"],
                      "all_possible_name":["wildtype","A1B","A0B","A0B/A1B"],
                      "n_mutations":[0,1,1],
                      "mutations":[["A", "B"],
                                   ["A", "B"]],
                      "length":2,
                      "n":3,
                      "mutant":"BB"})

    return test_data
