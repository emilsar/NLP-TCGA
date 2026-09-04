"""Survey a tree of Jupyter notebooks without opening any of them.

For every .ipynb found under a directory, prints its markdown headings and the
import lines from its code cells. That is usually enough to see how an
unfamiliar notebook codebase is organised and what it depends on.

Written to survey the guilopgar reference repo; works on any notebook tree.

Usage:
    python scan_nbs.py [directory]

    # default: the guilopgar reference repo, if it is cloned beside this one
    python scan_nbs.py
    python scan_nbs.py ../guilopgar/code
    python scan_nbs.py . > survey.txt
"""

import argparse
import json
import os
import sys

# Where the guilopgar reference notebooks live when cloned per repos.json.
DEFAULT_BASE = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "guilopgar", "code")
)

# Only the first N lines of a code cell are scanned for imports; imports below
# that in a cell are effectively never real imports.
CODE_HEAD_LINES = 35
MAX_WIDTH = 160


def scan(base):
    notebooks = []
    for root, _, files in os.walk(base):
        for name in files:
            if name.endswith(".ipynb") and ".ipynb_checkpoints" not in root:
                notebooks.append(os.path.join(root, name))
    notebooks.sort()

    if not notebooks:
        print("No notebooks found under %s" % base, file=sys.stderr)
        return 1

    for path in notebooks:
        print("\n========== " + os.path.relpath(path, base) + " ==========")
        try:
            with open(path, encoding="utf-8") as fh:
                nb = json.load(fh)
        except (OSError, ValueError) as exc:
            print("  !! could not read: %s" % exc)
            continue

        for cell in nb.get("cells", []):
            source = "".join(cell.get("source", []))
            if cell.get("cell_type") == "markdown":
                for line in source.splitlines():
                    if line.strip().startswith("#"):
                        print("  MD: " + line.strip()[:MAX_WIDTH])
            else:
                for line in source.splitlines()[:CODE_HEAD_LINES]:
                    stripped = line.strip()
                    if stripped.startswith(("import ", "from ")):
                        print("  IMP: " + stripped[:MAX_WIDTH])

    print("\n%d notebook(s) under %s" % (len(notebooks), base), file=sys.stderr)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base", nargs="?", default=DEFAULT_BASE,
                        help="directory to search (default: the guilopgar reference repo)")
    args = parser.parse_args()

    if not os.path.isdir(args.base):
        print("Not a directory: %s" % args.base, file=sys.stderr)
        return 2
    return scan(args.base)


if __name__ == "__main__":
    sys.exit(main())
