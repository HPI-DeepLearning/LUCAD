import os
import logging
import mxnet as mx


class MultiRunLoader(object):
    def __init__(self, prefix):
        self.prefix = prefix
        if "{val_subset}" not in self.prefix:
            raise RuntimeError("{val_subset} placeholder missing.")
        self.runs = {}
        for i in range(0, 10):
            loader = RunLoader(self.prefix.format(val_subset=i), quiet = True)
            if loader.is_valid():
                self.runs[i] = loader
        logging.info("Loaded {n} runs for {p}".format(n=len(self.runs), p=self.prefix))

    def load(self, subset, epoch="LAST"):
        if subset not in self.runs:
            raise RuntimeError("invalid subset")
        return self.runs[subset].load(epoch)


class RunLoader(object):
    def __init__(self, prefix, quiet = False):
        self.prefix = prefix
        self.valid = True

        if not quiet:
            logging.debug("Loading run %s" % self.prefix)

        self.directory = os.path.dirname(prefix)
        self.base_prefix = os.path.basename(prefix)

        self.symbol = "%s-symbol.json" % self.prefix
        if not os.path.isfile(self.symbol):
            self.valid = False

        self.epochs = filter(lambda x: x.startswith(self.base_prefix) and ".params" in x, os.listdir(self.directory))
        self.epochs = [e.replace("%s-" % self.base_prefix, "") for e in self.epochs]
        self.epochs = [int(e.replace(".params", "").replace("%s-" % self.base_prefix, "")) for e in self.epochs]

        if len(self.epochs) == 0:
            self.valid = False

        self.log = "%s.log" % self.prefix
        if not os.path.isfile(self.log):
            self.log = None

        if not quiet:
            logging.debug("Valid:  %s" % str(self.valid))
            logging.debug("Epochs: %s" % str(self.epochs))
            logging.debug("Log:    %s" % str(self.log))

    def check_validity(self):
        if not self.valid:
            raise RuntimeError("Cannot access incomplete runs.")

    def is_valid(self):
        return self.valid

    def epochs(self):
        self.check_validity()
        return self.epochs

    def log(self):
        self.check_validity()
        return self.log

    def log_file(self):
        self.check_validity()
        return open(self.log, "r")

    def results(self, epoch = "LAST"):
        self.check_validity()
        # Todo: implement
        return 0.0, 0.0

    def load(self, epoch = "LAST"):
        self.check_validity()
        if epoch == "LAST":
            epoch = max(self.epochs)
        return mx.model.load_checkpoint(self.prefix, epoch) + (epoch,)



