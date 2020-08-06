import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--remap", action="store_true")
parser.add_argument("--sanity", action="store_true")
parser.add_argument("--depth", type=int, default=2)
parser.add_argument("--width", type=int, default=64)
parser.add_argument("--lr", type=float, default=0.0025)
