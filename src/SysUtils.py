import pathlib as plb
import datetime
import sys
import argparse

def is_Win32():
    return sys.platform.startswith("win")

def make_dir(dir_path, exist_ok=True):
    plb.Path(dir_path).mkdir(parents=True, exist_ok=exist_ok)

def get_current_time_string():
    return datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')

def get_items(folder, what_to_get=None):
    if what_to_get is None:
        what_to_get = "stem"
    if what_to_get not in ["stem", "name", "full"]:
        raise ValueError("Incorrect flag")
    path = plb.Path(folder)
    if what_to_get == "stem":
        return [item.stem for item in path.iterdir()]
    elif what_to_get == "name":
        return [item.name for item in path.iterdir()]
    elif what_to_get == "full":
        return [str(item) for item in path.iterdir()]

def get_path(args):
    return "/".join(args)
    

def get_filename(path):
    return plb.Path(path).stem

def get_extension(path):
    return plb.Path(path).suffix

def get_parent_dir(path):
    return plb.Path(path).parent

def read_learning_rate(filename):
    if not plb.Path(filename).exists():
        return None
    lr = []
    with open(filename, "r") as f:
        for line in f:
            tokens = line.split()
            if len(tokens) > 2:
                raise ValueError("there are more than 2 entries")
            if len(tokens) == 0:
                continue
            elif len(tokens) == 1:
                lr += [float(tokens[0])]
            elif len(tokens) == 2:
                lr += [float(tokens[0])]*int(tokens[1])
    return lr


class ArgParser(object):
    def __init__(self):
        self.config = {}
        self.config["num_epochs"] = 300
        self.config["minibatch_size"] = 128
        self.config["epoch_size"] = 50000
        self.config["constlr"] = True
        self.config["lr"] = 0.0001
        self.config["lr_list"] = None
        self.config["momentum"] = 0.9

    def prepare(self):
        self.parser = argparse.ArgumentParser(description="Process input arguments for training")
        self.parser.add_argument("--epoch", type=int, default=self.config["num_epochs"])
        self.parser.add_argument("--minibatch", type=int, default=self.config["minibatch_size"])
        self.parser.add_argument("--epoch_size", type=int, default=self.config["epoch_size"])
        self.parser.add_argument("--constlr", action="store_true")
        self.parser.add_argument("--lr", type=float, default=self.config["lr"])
        self.parser.add_argument("--momentum", type=float, default=self.config["momentum"])
        self.parser.add_argument("--lr_file", type=str)

    def parse(self):
        self.args = self.parser.parse_args()
        self.config["num_epochs"] = self.args.epoch
        self.config["minibatch_size"] = self.args.minibatch
        self.config["epoch_size"] = self.args.epoch_size
        self.config["constlr"] = self.args.constlr
        self.config["lr"] = self.args.lr
        self.config["momentum"] = self.args.momentum

        if self.config["num_epochs"] < 1 or self.config["minibatch_size"] < 1 or self.config["epoch_size"] < 1:
            raise ValueError("number of minibatches, epochs or minibatch size must be positive")

        if (not self.config["constlr"]) and self.args.lr_file:
            self.config["lr_list"] = read_learning_rate(self.args.lr_file)
