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
    out_csv = csv.writer(args.output)

    files = [open(os.path.join(r, "concat.csv"), "rb") for r in args.result]
    readers = [csv.reader(f) for f in files]
    names = [os.path.basename(r) for r in args.result]

    headers = []

    for r in readers:
        headers.append(r.next())

    assert all_equal(headers)

    output_header = filter(lambda col: "prediction" != col and "probability" != col, headers[0])

    out_prob_index = len(output_header)
    output_header.extend(["probability_%s" % n for n in names])

    other_prob_cols = [
        # [key, accum_function, filter_function]
        # ["avg_stage_A", avg, lambda col_name: any(c in col_name for c in ["A", "B", "C"])],
        # ["avg_stage_B", avg, lambda col_name: any(c in col_name for c in ["T", "U", "V", "W", "X", "Y", "Z"])],
        # ["avg_all", avg, lambda _: True],
        # ["max_stage_A", max, lambda col_name: any(c in col_name for c in ["A", "B", "C"])],
        # ["max_stage_B", max, lambda col_name: any(c in col_name for c in ["T", "U", "V", "W", "X", "Y", "Z"])],
        # ["max_all", max, lambda _: True],
        # ["min_stage_A", min, lambda col_name: any(c in col_name for c in ["A", "B", "C"])],
        # ["min_stage_B", min, lambda col_name: any(c in col_name for c in ["T", "U", "V", "W", "X", "Y", "Z"])],
        # ["min_all", min, lambda _: True],
        ["avg_stage_A+", avg, lambda col_name: any(c in col_name for c in ["A", "B", "C", "D"])],
        ["max_stage_A+", max, lambda col_name: any(c in col_name for c in ["A", "B", "C", "D"])],
        ["min_stage_A+", min, lambda col_name: any(c in col_name for c in ["A", "B", "C", "D"])],
    ]

    other_prob_index = len(output_header)
    for key, _, _ in other_prob_cols:
        output_header.append("probability_%s" % key)

    in_prob_col = headers[0].index("probability")

    out_csv.writerow(output_header)

    reindex = {headers[0].index(col): output_header.index(col) for col in output_header if col in headers[0]}
    try:
        while True:
            line = [0] * len(output_header)
            for i, r in enumerate(readers):
                row = r.next()
                if i == 0:
                    for index_in, index_out in reindex.items():
                        line[index_out] = row[index_in]
                line[out_prob_index + i] = row[in_prob_col]

            for i, triple in enumerate(other_prob_cols):
                key, accum_fn, filter_fn = triple
                probs = [float(x) for x in line[out_prob_index:out_prob_index + len(readers)]]
                filtered = []
                for n, p in zip(names, probs):
                    if filter_fn(n):
                        filtered.append(p)
                line[other_prob_index + i] = accum_fn(filtered)

            out_csv.writerow(line)
    except StopIteration:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=str, nargs="+", help="result folders")
    parser.add_argument("--output", type=argparse.FileType("w", 0), help="where to write the output, default: stdout", default=sys.stdout)

    main(parser.parse_args())
