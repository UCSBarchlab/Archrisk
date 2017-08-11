import collections
import re
from sympy import symbols, numbered_symbols, IndexedBase, Idx, Function
from sympy.parsing.sympy_parser import parse_expr, convert_equals_signs

class SympyHelper(object):

    @staticmethod
    def initSyms(syms):
        sym_dict = {}
        for sym in syms:
            sym_dict[sym] = symbols(sym)
        return sym_dict

    @staticmethod
    def initFuncs(funcs):
        func_dict = {}
        for func in funcs:
            func_dict[func] = symbols(func, cls=Function)
        return func_dict

    @staticmethod
    def initExprs(exprs, syms):
        expr_list = []
        for expr in exprs:
            parsed_expr = parse_expr(expr, local_dict=syms,
                    transformations=(convert_equals_signs,))
            expr_list.append(parsed_expr)
        return expr_list
