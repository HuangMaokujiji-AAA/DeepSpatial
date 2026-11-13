#!/usr/bin/env python3
"""Inspect/import the published atlas annotation table without installing the project."""
import argparse
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, default=PROJECT/'resources/markers.csv')
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--sheet')
    parser.add_argument('--header', type=int, default=0)
    parser.add_argument('--subclass-column')
    parser.add_argument('--marker-columns', nargs='+')
    args = parser.parse_args()
    import pandas as pd
    if args.inspect:
        if args.input.suffix.lower() == '.csv':
            print(pd.read_csv(args.input, nrows=5).to_string(index=False))
        else:
            book = pd.ExcelFile(args.input)
            for sheet in book.sheet_names:
                print(f'SHEET {sheet!r}')
                print(pd.read_excel(book, sheet_name=sheet, header=args.header, nrows=5).to_string(index=False))
        return
    from raspatial.markers import import_table
    sheet = int(args.sheet) if args.sheet is not None and args.sheet.isdigit() else args.sheet
    table = import_table(args.input, args.output, sheet, args.subclass_column, args.marker_columns, args.header)
    print(f'Imported {len(table):,} marker records across {table.subclass.nunique():,} subclass labels: {args.output}')


if __name__ == '__main__':
    main()
