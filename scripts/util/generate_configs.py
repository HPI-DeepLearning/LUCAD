import os
import ConfigParser
from collections import OrderedDict


def set_model_values(template, path, weight, index):
    template.set("models", "model%d" % index, os.path.join("../runs", str(path)))
    template.set("models", "weight%d" % index, str(weight))
    template.set("models", "epoch%d" % index, "LAST")


if __name__ == "__main__":
    FOLDER_FORMAT = "2017-10-18_%s"
    FILE_FORMAT = "generated_config_%s.ini"

    stageA = OrderedDict(sorted({
        "A": "v2_downsampledC/leakyrelu_with_s10_val{val_subset}",
        "B": "leakyrelunet-v2_dice_memmap_shuffled/leakyrelu_with_s10_val{val_subset}",
        "C": "leakyrelunet-v2_kokA/leakyrelu_with_s10-val{val_subset}",
    }.items()))

    stageB = OrderedDict(sorted({
        "T": "simplenet-non-augmented/leakyrelu_with_s10_val{val_subset}",
        "U": "v2_downsampledA/leakyrelu_with_s10_val{val_subset}",
        "V": "v2_xyD/leakyrelu_with_s10_val{val_subset}",
        "W": "leakyrelunet-v2_xyC/leakyrelu_with_s10_val{val_subset}",
        "X": "v2_xyE/leakyrelu_with_s10_val{val_subset}",
        "Y": "leakyrelunet-v2_fonova7/leakyrelu_with_s10_val{val_subset}",
        "Z": "v2_fonova7_high_res/leakyrelu_with_s10_val{val_subset}",
    }.items()))

    i = 0

    WEIGHTS = [(7.0, 3.0), (1.0, 1.0), (3.0, 7.0)]

    for p in stageA:
        key = "%03d_%s" % (i, p)

        parser = ConfigParser.SafeConfigParser()
        parser.read(os.path.join("config", "ms-config.ini.default"))

        set_model_values(parser, stageA[p], 1.0, 0)
        parser.set("output", "dir", FOLDER_FORMAT % key)
        parser.set("mxnet", "batch_size", "128")

        parser.write(open(os.path.join("config", FILE_FORMAT % (key)), "w"))
        i += 1

    for p in stageB:
        key = "%03d_%s" % (i, p)

        parser = ConfigParser.SafeConfigParser()
        parser.read(os.path.join("config", "ms-config.ini.default"))

        set_model_values(parser, stageB[p], 1.0, 0)
        parser.set("output", "dir", FOLDER_FORMAT % key)
        parser.set("mxnet", "batch_size", "128")

        parser.write(open(os.path.join("config", FILE_FORMAT % (key)), "w"))
        i += 1

    for p1 in stageA:
        for p2 in stageB:
            for w1, w2 in WEIGHTS:
                key = "%03d_%d-%s+%d-%s" % (i, round(w1), p1, round(w2), p2)

                parser = ConfigParser.SafeConfigParser()
                parser.read(os.path.join("config", "ms-config.ini.default"))

                set_model_values(parser, stageA[p1], w1, 0)
                set_model_values(parser, stageB[p2], w2, 1)
                parser.set("output", "dir", FOLDER_FORMAT % key)
                parser.set("mxnet", "batch_size", "128")

                parser.write(open(os.path.join("config", FILE_FORMAT % (key)), "w"))
                i += 1
