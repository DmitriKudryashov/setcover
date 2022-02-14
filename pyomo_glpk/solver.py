#!/usr/bin/python
# -*- coding: utf-8 -*-

# The MIT License (MIT)
#
# Copyright (c) 2014 Carleton Coffrin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


from collections import namedtuple
import pyomo.environ as pyo
import pandas as pd
import numpy as np

Set = namedtuple("Set", ['index', 'cost', 'items'])


def get_df(item_count, set_count, sets):
    initial_cols = ['id', 'cost', 'cost_per_item', 'total_items']
    additional_cols = list(map(str, list(range(item_count))))

    df = pd.DataFrame(np.zeros((set_count, len(initial_cols + additional_cols)), dtype='i'),
                      columns=(initial_cols + additional_cols))
    df['cost'] = df['cost'].astype(float)
    df['cost_per_item'] = df['cost_per_item'].astype(float)

    for i in range(len(sets)):
        coverage_list = map(str, list(sets[i].items))
        df.loc[i, 'id'] = sets[i].index
        df.loc[i, 'cost'] = sets[i].cost
        for item in coverage_list:
            df.loc[i, item] = 1

    df['total_items'] = df.iloc[:, 4:].sum(axis=1)
    df['cost_per_item'] = df['cost'] / df['total_items']

    return df


def get_coordinates(df):
    coverage_dict = {}

    for idx, row in df.iterrows():
        for col in row.index:
            # print (idx, col, row[col])
            coverage_dict[(idx, col)] = row[col]

    return coverage_dict


def optimize(df):
    col_index = df.columns[4:].to_list()
    col_base = df.columns[0:4].to_list()

    # remove zero rows and columns
    df = df.loc[
        (df[col_index].loc[
            df[col_index].sum(axis='columns') > 0]).index]

    col_index = df[
        (df[col_index].sum(axis='index').loc[
            df[col_index].sum(axis='index') > 0]).index
    ].columns.to_list()

    # create model
    model = pyo.ConcreteModel()

    # firestation index
    model.i = pyo.Set(initialize=df.index.to_list())
    # region index
    model.j = pyo.Set(initialize=col_index)

    # cost data
    model.cost = pyo.Param(model.i, initialize=df[['cost']].to_dict()['cost'])

    # coverage matrix
    model.coverage = pyo.Param(model.i, model.j,
                               initialize=get_coordinates(df[col_index]))

    # main variable to choose firestations
    model.x = pyo.Var(model.i, within=pyo.Binary, initialize=0)

    # auxilary variable for objective function
    model.OF = pyo.Var()

    # constrain to provide full coverage
    def rule_C1(model, j):
        return sum(model.x[i] * model.coverage[i, j] for i in model.i) >= 1

    model.C1 = pyo.Constraint(model.j, rule=rule_C1)

    # constraint for OF
    def rule_C2(model):
        return sum(model.x[i] * model.cost[i] for i in model.i) == model.OF

    model.C2 = pyo.Constraint(rule=rule_C2)

    # objective function
    def obj_rule(model):
        return sum(model.x[i] * model.cost[i] for i in model.i)

    model.obj = pyo.Objective(expr=model.OF, sense=pyo.minimize)

    # solver
    solver = pyo.SolverFactory('glpk')

    solver.solve(model, timelimit=1800)

    return model.x.get_values()


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')
    parts = lines[0].split()
    item_count = int(parts[0])
    set_count = int(parts[1])

    sets = []
    for i in range(1, set_count + 1):
        parts = lines[i].split()
        sets.append(Set(i - 1, float(parts[0]), map(int, parts[1:])))

    raw_df = get_df(item_count, set_count, sets)

    keys = set()

    # optimization
    x_dict = optimize(raw_df)
    keys = keys.union(set([k for k, v in x_dict.items() if v == 1]))

    # result recording
    raw_df['x'] = 0
    raw_df.loc[keys, 'x'] = 1

    # results
    obj = raw_df[['x', 'cost']].product(axis='columns').sum()
    solution = raw_df['x'].to_list()

    # prepare the solution in the specified output format
    output_data = str(obj) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/sc_6_1)')