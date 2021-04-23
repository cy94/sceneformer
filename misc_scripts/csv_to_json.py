"""
Convert a csv/space separated file
a b
c d

to a JSON

{
    a: b,
    c: d
}
"""
import argparse
import csv
import json


def main(args):
    j = {}
    with open(args.csv_path) as f:
        lines = [l.split() for l in f.readlines()]
    for (key, val) in lines:
        try:
            j[key] = int(val)
        except ValueError:
            j[key] = val

    with open(args.json_path, "w") as f:
        json.dump(j, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("json_path")
    args = parser.parse_args()
    main(args)
