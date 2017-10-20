import argparse
import os
import sys
import csv

from util import helper


def all_equal(lst):
    return lst[1:] == lst[:-1]


def avg(values):
    return sum(values) / len(values)


def main(args):
    in_csv = csv.reader(args.input)

    header = in_csv.next()

    combinations = []
    for t in args.threshold:
        for col in args.column:
            combinations.append((col, t))

    dirs = [os.path.join(args.output_folder, "%s_%.2f" % (col, t)) for col, t in combinations]
    for d in dirs:
        if not os.path.isdir(d):
            os.makedirs(d)

    files = [open(os.path.join(args.output_folder, "%s_%.2f" % (col, t), "concat.csv"), "wb") for col, t in combinations]
    writers = [csv.writer(f) for f in files]
    idx = [header.index(col) for col, _ in combinations]

    output_header = ["seriesuid", "coordX", "coordY", "coordZ", "probability", "class", "prediction"]

    reindex = {header.index(col): output_header.index(col) for col in output_header if col in header}
    prob_idx = output_header.index("probability")
    prediction_idx = output_header.index("prediction")

    for i, comb in enumerate(combinations):
        writers[i].writerow(output_header)

    for row in in_csv:
        line = [0] * len(output_header)

        for index_in, index_out in reindex.items():
            line[index_out] = row[index_in]

        for i, comb in enumerate(combinations):
            _, threshold = comb
            line[prob_idx] = row[idx[i]]
            line[prediction_idx] = 0 if row[idx[i]] < threshold else 1
            writers[i].writerow(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=argparse.FileType("rb", 0), help="collated input csv file, default: stdin", default=sys.stdin)
    parser.add_argument("--column", type=str, nargs="+", help="which columns to split into separate files")
    parser.add_argument("--threshold", type=float, nargs="+", help="which minimum probability is needed for a sample to be classified as positive", default=(0.5, ))
    parser.add_argument("--output-folder", type=str, help="where to write the output files", default="../results/split")

    main(parser.parse_args())
