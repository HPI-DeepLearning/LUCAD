import os

from util import helper
from candidate_iterator import CandidateIter
from distributed_iterator import DistributedIter


def get_iterator(root, subsets, batch_size, chunk_size = 100, shuffle = False, prefetch = False):
    info = helper.read_info_file(os.path.join(root, "subset%d" % subsets[0], "info.txt"))
    if "type" not in info or info["type"] == "CandidateStorage":
        return CandidateIter(root, subsets, batch_size = batch_size, shuffle = shuffle, chunk_size = chunk_size)
    if info["type"] == "DistributedStorage":
        return DistributedIter(root, subsets, batch_size = batch_size, prefetch = prefetch, shuffle = shuffle)
    return None
