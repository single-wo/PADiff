from collections import Counter
from scipy import spatial as sci_spatial
import numpy as np


ATOM_TYPE_DISTRIBUTION = {
    6: 0.6715020339893559,
    7: 0.11703509510732567,
    8: 0.16956379168491933,
    9: 0.01307879304486639,
    15: 0.01113716146426898,
    16: 0.01123926340861198,
    17: 0.006443861300651673,
}


def eval_atom_type_distribution(pred_counter: Counter):
    total_num_atoms = sum(pred_counter.values())
    pred_atom_distribution = {}
    for k in ATOM_TYPE_DISTRIBUTION:
        pred_atom_distribution[k] = pred_counter[k] / total_num_atoms
    js = sci_spatial.distance.jensenshannon(np.array(list(ATOM_TYPE_DISTRIBUTION.values())),
                                            np.array(list(pred_atom_distribution.values())))
    return js
