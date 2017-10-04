import argparse
import os
import sys

from util import helper


class AbstractFormatter(object):
    def __init__(self, output):
        self.output = output

    def print_data(self, header, values):
        raise RuntimeError("Should be implemented by subclass")


class CSVFormatter(AbstractFormatter):
    def print_data(self, header, values):
        self.output.write("%s\n" % ",".join(header))
        for row in values:
            self.output.write("%s\n" % ",".join(row))


class WikiFormatter(AbstractFormatter):
    def print_data(self, header, values):
        raise RuntimeError("Should be implemented by subclass")


def get_data(root):
    subsets = helper.get_filtered_subsets(root)
    if len(subsets) == 0:
        return {}

    info_files = {}
    for subset in subsets:
        info_files[subset] = helper.read_info_file(os.path.join(root, subset, "info.txt"))

    combined_info = helper.check_and_combine(info_files)
    combined_info["started"] = info_files[subsets[0]]["started"]
    if "finished" in info_files[subsets[0]]:
        combined_info["finished"] = "True"
    else:
        combined_info["finished"] = "False"

    return combined_info


def main(args):
    datasets = os.listdir(args.root)

    if args.wiki:
        formatter = WikiFormatter(args.output)
    else:
        formatter = CSVFormatter(args.output)

    all_data = []
    for ds in datasets:
        if "CSVFILES" in os.listdir(os.path.join(args.root, ds)):
            continue
        data = get_data(os.path.join(args.root, ds))
        data["folder"] = ds
        all_data.append(data)

    all_data.sort(key = lambda x: x["started"], reverse = True)

    all_print_data = []
    for data in all_data:
        if "augmentation" not in data:
            data["augmentation"] = "dice"
        if "shuffled" not in data:
            if "shuffled" in data["folder"]:
                data["shuffled"] = "True"
            else:
                data["shuffled"] = "False"
        if "type" not in data:
            data["type"] = "CandidateStorage"

        print_data = [data[c] for c in args.columns]

        all_print_data.append(print_data)

    formatter.print_data(args.columns, all_print_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=str, help="containing all datasets")
    parser.add_argument("--output", type=argparse.FileType("w", 0), help="where to write the output, default: stdout", default=sys.stdout)
    parser.add_argument("--columns", help="which columns to write out", default=["folder", "augmentation", "shuffled", "type", "finished"])
    parser.add_argument("--exceptions", help="except a few datasets", default=["test"])
    parser.add_argument("--sort-by", help="sort by which column", default="started")
    parser.add_argument("--wiki", help="produce markdown style output", action="store_true")

    main(parser.parse_args())
