import os
import ConfigParser


def set_model_values(template, path, weight, index):
    template.set("models", "model%d" % index, str(path))
    template.set("models", "weight%d" % index, str(weight))
    template.set("models", "epoch%d" % index, "AUTO")


if __name__ == "__main__":
    prefixes = {
        "A": "leakyrelunet-v2_dice_memmap_shuffled/leakyrelu_with_s10_val",
        "B": "simplenet-non-augmented/leakyrelu_with_s10_val",
        "C": "simplenet-v2_dice_memmap_shuffled/test-val",
        "D": "v2_downsampledC/leakyrelu_with_s10_val",
        "E": "leakyrelunet-v2_kokA/leakyrelu_with_s10-val",
        "F": "leakyrelunet-v2_xyC/leakyrelu_with_s10_val",
    }

    weights = [(1.0, 1.0), (3.0, 7.0)]

    i = 0

    for p in prefixes:
        key = p

        parser = ConfigParser.SafeConfigParser()
        parser.read(os.path.join("config", "ms-config.ini.default"))

        set_model_values(parser, prefixes[p], 1.0, 0)
        parser.set("output", "dir", "{timestamp} %s" % key)

        parser.write(open(os.path.join("config", "generated_config_%d_%s.ini" % (i, key)), "w"))
        i += 1

    for p1 in prefixes:
        for p2 in prefixes:
            if p1 == p2:
                continue
            for w1, w2 in weights:
                key = "%d-%s+%d-%s" % (round(w1), p1, round(w2), p2)

                parser = ConfigParser.SafeConfigParser()
                parser.read(os.path.join("config", "ms-config.ini.default"))

                set_model_values(parser, prefixes[p1], w1, 0)
                set_model_values(parser, prefixes[p2], w2, 0)
                parser.set("output", "dir", "{timestamp} %s" % key)

                parser.write(open(os.path.join("config", "generated_config_%d_%s.ini" % (i, key)), "w"))
                i += 1
