import os


def generate_intervals(blocksize, total_snps):
    if blocksize is None:
        blocksize = total_snps
    intervals = range(0, total_snps, blocksize)

    if intervals[-1] != total_snps:
        intervals.append(total_snps)
    return intervals


def ensure_dir(f):
    d = os.path.dirname(f)

    # d is '' when f is file is in current directory e.g. f = "file.txt"
    if d and not os.path.exists(d):
        os.makedirs(d)