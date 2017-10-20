import argparse
import os
import sys
import csv

from util import helper


FP_RATES = (0.125, 0.25, 0.5, 1, 2, 4, 8)
VALUES = ("Sensivity[Mean]", "Sensivity[Lower bound]", "Sensivity[Upper bound]")
COLUMNS = ("name", "fp_average", "fp_rate") + VALUES


def find_after_colon(row):
    return row.find(': ') + 2, len(row)


class Result(object):
    def __init__(self, path):
        self.path = path
        self.collect_froc()
        # self.collect_scoring()

    def collect_froc(self):
        self.csv_file = open(os.path.join(self.path, "CADEvaluation", "froc_concat_bootstrapping.csv"), "rb")
        self.reader = csv.reader(self.csv_file)
        self.header = self.reader.next()
        self.index = {col: self.header.index(col) for col in self.header}
        self.name = os.path.basename(self.path)

        self.values = []

        sum_ = 0.0

        for row in self.reader:
            for fp_rate in FP_RATES:
                if abs(float(row[self.index["FPrate"]]) - fp_rate) < 0.0005:
                    column_values = {"fp_rate": str(fp_rate)}
                    for val in VALUES:
                        column_values[val] = row[self.index[val]]
                    sum_ += float(column_values["Sensivity[Mean]"])
                    self.values.append(column_values)
        assert len(FP_RATES) == len(self.values)
        self.average = sum_ / len(FP_RATES)

    def collect_scoring(self):
        self.log_file = open(os.path.join(self.path, "scoring.log"), "r")

        retrieval = [
            ["Subsets for iterator", "subset", lambda x: (x.find('[\'') + 2, x.find('\']'))],
            ["Precision", "precision", find_after_colon],
            ["Recall", "recall", find_after_colon],
            ["F1-total", "f1", find_after_colon],
        ]

        self.scoring = []

        current = {}

        for row in self.log_file:
            for key, col, fn in retrieval:
                if key in row:
                    if col == "subset":
                        self.scoring.append(current)
                        current = {}
                    s, e = fn(row)
                    current[col] = row[s:e]
        self.scoring.append(current)

    def write_to(self, csv_out, header):
        for values in self.values:
            row = []
            for col in header:
                if col in values:
                    row.append(values[col])
                    continue
                if col == "fp_average":
                    row.append(self.average)
                    continue
                if col == "name":
                    row.append(self.name)
                    continue
                raise RuntimeError("Unknown column!")
            csv_out.writerow(row)


def main(args):
    out_csv = csv.writer(args.output)

    results = []
    for r in args.result:
        results.append(Result(r))

    out_csv.writerow([col.replace("[", "_").replace("]", "_").replace(" ", "_") for col in COLUMNS])
    for r in results:
        r.write_to(out_csv, COLUMNS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=str, nargs="+", help="result folders")
    parser.add_argument("--output", type=argparse.FileType("w", 0), help="where to write the output, default: stdout", default=sys.stdout)

    main(parser.parse_args())
