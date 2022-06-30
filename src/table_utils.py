# Code adapted from Ciwan Ceylan
from typing import Iterable

import numpy as np


def float_exponent_notation(float_number, precision_digits, format_type="g"):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with `precision_digits` digits of
    mantissa precision, printing a normal decimal if an
    exponent isn't necessary.
    """
    e_float = "{0:.{1:d}{2:}}".format(float_number, precision_digits, format_type)
    if "e" not in e_float:
        return "{}".format(e_float)
    mantissa, exponent = e_float.split("e")
    cleaned_exponent = exponent.strip("+").lstrip("0")
    return "{0} \\times 10^{{{1}}}".format(mantissa, cleaned_exponent)


def create_table_start(row_columns, other_cols):
    layout = "{" + f"{len(row_columns) * 'r' + len(other_cols) * 'c'}" + "}"
    out = "\\begin{{tabular}}{layout}".format(layout=layout)
    out = "\n".join([out, "\\toprule", " & ".join(row_columns + other_cols) + "\\\\", "\\midrule\n"])
    return out


def create_table_end():
    return "\\bottomrule\n\\end{tabular}"


def get_best_threshold(grouped, highorlow="low", num_sigma=1):
    best_val_threshold = {}
    mean_cols = [c for c in grouped.columns if c[1] == "mean"]
    for row in grouped.index:
        vals = grouped.loc[row, mean_cols]
        if highorlow == "low":
            col = vals.index[vals.argmin()][0]
            thres = grouped.loc[row, (col, "mean")] + num_sigma * grouped.loc[row, (col, "std")]
            best_val_threshold[row] = (thres, False)
        else:
            col = vals.index[vals.argmax()][0]
            thres = grouped.loc[row, (col, "mean")] - num_sigma * grouped.loc[row, (col, "std")]
            best_val_threshold[row] = (thres, True)

    return best_val_threshold


def tbl_elm(value, std, is_best, num_decimal=3):
    element = f"{float_exponent_notation(value, num_decimal)} \\pm {float_exponent_notation(std, num_decimal)}"
    # element = float_exponent_notation(value, num_decimal)
    element = "$\\mathbf{}$".format("{" + element + "}") if is_best else "${}$".format(element)
    return element


def tbl_elm_no_std(value, is_best, num_decimal=2):
    element = f"{np.round(value, decimals=num_decimal):.{num_decimal}f}"
    element = "$\\mathbf{}$".format("{" + element + "}") if is_best else "${}$".format(element)
    return element


def create_tbl_row(grouped, row_index, columns, thres):
    elements = []
    for col in columns:
        val = grouped.loc[row_index, (col, "mean")]
        std = grouped.loc[row_index, (col, "std")]
        elements.append(
            tbl_elm(val, std, val >= thres[row_index][0] if thres[row_index][1] else val <= thres[row_index][0]))
    if isinstance(row_index, str) or isinstance(row_index, int):
        row_index = [str(row_index)]
    elif isinstance(row_index, Iterable):
        row_index = list(str(r) for r in row_index)
    out = " & ".join(row_index + elements)
    out += "\\\\\n"
    return out


def make_table(df, row_columns, other_columns, ordering=None, highorlow="low"):
    if isinstance(row_columns, dict):
        df = df.rename(columns=row_columns)
        row_columns = list(row_columns.values())
    if isinstance(other_columns, dict):
        df = df.rename(columns=other_columns)
        other_columns = list(other_columns.values())

    grouped = df.loc[:, row_columns + other_columns].groupby(row_columns).agg(["mean", "std"])
    thres = get_best_threshold(grouped, highorlow=highorlow)
    print(thres)
    out = create_table_start(row_columns, other_columns)
    grouped = grouped.reindex(ordering, copy=False)
    for row in grouped.index:
        out += create_tbl_row(grouped, row, other_columns, thres)

    out += create_table_end()
    return grouped, out
