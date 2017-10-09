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
        max_lengths = []
        shorten = {}
        for i in range(0, len(header)):
            max_l = 0
            for row in values:
                if len(row[i]) > max_l:
                    max_l = len(row[i])
            if len(header[i]) > max_l:
                key = len(shorten) + 1
                shorten[key] = header[i]
                header[i] = "%s[^%d]" % (header[i][:max(1, max_l-2)], key)
                max_l = max(max_l, len(header[i]))
            max_lengths.append(max_l)

        self.output.write("# Prepared datasets\n\n")
        self.output.write("|%s|\n" % "|".join(self.format(header, max_lengths)))
        self.output.write("|%s|\n" % "|".join(self.separator(max_lengths)))
        for row in values:
            self.output.write("|%s|\n" % "|".join(self.format(row, max_lengths)))
        self.output.write("\n(table was generated at %s)\n" % helper.now())
        self.output.write("\n%s\n" % "\n".join(["[^%s]: %s" % l for l in shorten.items()]))

    @staticmethod
    def format(row, max_lengths):
        return [v.ljust(max_lengths[i]) for i, v in enumerate(row)]

    @staticmethod
    def separator(max_lengths):
        fn = [":".rjust if i == 0 else ":".ljust for i in range(0, len(max_lengths))]
        return [fn[i](j, "-") for i, j in enumerate(max_lengths)]


def to_array(raw):
    if isinstance(raw, list):
        return raw

    values = [x.strip() for x in raw.replace(",)", ")").replace(",]", "")[1:-1].replace("'", "").replace('"', "").split(",")]
    try:
        values = [float(v) for v in values]
    except ValueError:
        pass
    return values


def calc_factor(info):
    flip = len(to_array(info["flip"]))
    resize = len(to_array(info["resize"]))
    rotate = {"none": 1, "dice": 24, "xy": 4}[info["rotate"]]
    translations = int(info["translate"]) if "translate" in info else 1
    return flip * resize * rotate * translations


def get_data(root):
    subsets = helper.get_filtered_subsets(root)
    if len(subsets) == 0:
        return {}

    info_files = {}
    for subset in subsets:
        info_files[subset] = helper.read_info_file(os.path.join(root, subset, "info.txt"))

    combined_info = helper.check_and_combine(info_files)
    first_info = info_files[subsets[0]]

    combined_info["started"] = first_info["started"].replace(" ", "/")
    if "args" in first_info:
        transfer = ["factor", "ratio", "cubesize"]
        for t in transfer:
            if t in first_info["args"]:
                combined_info[t] = first_info["args"][t]

    if "factor" not in combined_info or combined_info["factor"] == "0":
        combined_info["factor"] = str(calc_factor(combined_info))
    else:
        combined_info["factor"] += " (rnd)"

    combined_info["resize"] = to_array(combined_info["resize"])
    res_array = to_array(combined_info["resize"])
    if len(combined_info["resize"]) > 4:
        values = tuple(res_array[:2] + res_array[-1:] + [len(res_array)])
        combined_info["resize"] = "%s, %s, ..., %s (%d)" % values
    else:
        combined_info["resize"] = res_array

    if "finished" in first_info:
        combined_info["finished"] = "True"
    else:
        combined_info["finished"] = "False"

    if "type" in combined_info:
        combined_info["type"] = combined_info["type"].replace("CandidateStorage", "memmap").replace("DistributedStorage", "array")

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

    hide = {
        "rotate": "none",
        "translate_limits": "[0, 0]",
        "translate": 1,
        "resize": [1.0],
        "flip": "('',)",
        "ratio": "-1"
    }

    all_print_data = []
    for data in all_data:
        print_data = []
        for c in args.columns:
            if c not in data or (c in hide and hide[c] == data[c]):
                print_data.append("-")
            else:
                print_data.append(str(data[c]).replace(" ", ""))

        all_print_data.append(print_data)

    formatter.print_data(args.columns, all_print_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=str, help="containing all datasets")
    parser.add_argument("--output", type=argparse.FileType("w", 0), help="where to write the output, default: stdout", default=sys.stdout)
    parser.add_argument("--columns", help="which columns to write out", default=["folder", "augmentation", "rotate", "translate_limits", "translate", "factor", "ratio", "resize", "flip", "shuffled", "type", "started"])
    parser.add_argument("--exceptions", help="except a few datasets", default=["test"])
    parser.add_argument("--sort-by", help="sort by which column", default="started")
    parser.add_argument("--wiki", help="produce markdown style output", action="store_true")

    main(parser.parse_args())
