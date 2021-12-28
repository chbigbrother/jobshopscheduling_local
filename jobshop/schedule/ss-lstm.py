#!/usr/bin/env python # -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from tensorflow.keras.utils import to_categorical, plot_model
from keras.utils import np_utils
from keras.models import Sequential,Model
from keras.layers import Input, Lambda, Conv1D, MaxPool1D,Dense,Flatten
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import (Conv2D, RepeatVector, Flatten, MaxPooling2D, LSTM,TimeDistributed)
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adadelta, Nadam
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from itertools import combinations, permutations
from ortools.constraint_solver import pywrapcp
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import keras, sys, datetime, progressbar
import pandas as pd
import copy, random, time


filedir = 'schedule/'

def ss_lstm(request):
    readFile = request.FILES['file'];
    read = pd.read_csv('./media/' + readFile.name, encoding='UTF8')

    raw_data = pd.read_csv("./training/2019_raw.csv")
    raw_data = raw_data[raw_data['가동율']>=90]
    raw_data = raw_data[raw_data['운행시간(HHMM)']<=1011]
    raw_data["time_len"]=raw_data['운행시간(HHMM)'].astype(str).str.len()
    raw_data = raw_data[raw_data['time_len']>=3] ##delete the length less than 2

    raw_data.loc[raw_data['time_len']==3,['운행시간[HHMM]']]=raw_data[raw_data['time_len']==3]['운행시간(HHMM)'].astype(str).str[0].astype(int)*60+raw_data[raw_data['time_len']==3]['운행시간(HHMM)'].astype(str).str[1:3].astype(int)
    raw_data.loc[raw_data["time_len"]==4,['운행시간(HHMM)']]=raw_data[raw_data['time_len']==4]['운행시간(HHMM)'].astype(str).str[0:2].astype(int)*60+raw_data[raw_data['time_len']==4]['운행시간(HHMM)'].astype(str).str[3:4].astype(int)
    raw_data['운행시간(HHMM)']=(raw_data['운행시간(HHMM)']/60).round(2) #to hour

    raw_data = raw_data.iloc[:, 0:11]

    name_product = raw_data['제품명'].unique()
    data=[]
    for name in name_product:
        sub_data=raw_data[raw_data['제품명']==name]
        data.append(sub_data)


    def new_variable_NameMachine(data):
        """data is one dataframe: groupby name, machineID
        """
        df1 = pd.DataFrame()
        df1 = data.groupby(['제품명', '작업 호기']).sum().iloc[:, [6]]
        df1["density"] = data["실제 기계 밀도"].unique()[0]
        df1["total processing time"] = data.groupby(['제품명', '작업 호기']).sum()["운행시간(HHMM)"]

        df1["machine numbers"] = data.groupby(['제품명', '작업 호기']).size()
        df1["rpm"] = data.groupby(['제품명', '작업 호기']).mean()["RPM"]
        df1["name of product"] = df1.index.levels[0][0]
        df1["machine"] = df1.index.levels[1]

        return df1


    def new_variable_NameTypeMachine(data):
        """data is one dataframe"""
        df1 = pd.DataFrame()
        df1 = data.groupby(['제품명', '작업 반', '작업 호기']).sum().iloc[:, [6]]
        df1["density"] = data["실제 기계 밀도"].unique()[0]
        df1["total processing time"] = data.groupby(['제품명', '작업 반', '작업 호기']).sum()["운행시간(HHMM)"]

        df1["machine numbers"] = data.groupby(['제품명', '작업 반', '작업 호기']).size()
        df1["rpm"] = data.groupby(['제품명', '작업 반', '작업 호기']).mean()["RPM"]
        df1["name of product"] = df1.index.levels[0][0]
        df1["machine"] = df1.index.get_level_values(2)
        return df1


    def new_variable_NameDateMachine(data):
        """data is one dataframe"""
        df1 = pd.DataFrame()
        df1 = data.groupby(['제품명', '작업일자', '작업 호기']).sum().iloc[:, [5]]
        df1["density"] = data["실제 기계 밀도"].unique()[0]
        df1["total processing time"] = data.groupby(['제품명', '작업일자', '작업 호기']).sum()["운행시간(HHMM)"]

        df1["machine numbers"] = data.groupby(['제품명', '작업일자', '작업 호기']).size()
        df1["rpm"] = data.groupby(['제품명', '작업일자', '작업 호기']).mean()["RPM"]
        df1["name of product"] = df1.index.levels[0][0]
        df1["machine"] = df1.index.get_level_values(2)
        return df1


    def create_new_dataframe(nameofProducts, raw_data, varGenerationFunction):
        myframe = pd.DataFrame()
        for name in nameofProducts:
            sub_data = raw_data[raw_data["제품명"] == name]
            myDataframe = varGenerationFunction(sub_data)
            myframe = myframe.append(myDataframe, ignore_index=True)
        return myframe

    df1=create_new_dataframe(name_product,raw_data,new_variable_NameMachine)
    df2=create_new_dataframe(name_product,raw_data,new_variable_NameTypeMachine)
    df3=create_new_dataframe(name_product,raw_data,new_variable_NameDateMachine)
    df=pd.concat([df1,df2,df3])
    df=pd.concat([df2,df3])
    df=df3

    def onehot_encoder(data):
        """data is the variabble need to convert into one-hot encoder"""
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(data.values)
        # binary encode
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        return onehot_encoded

    density_encoder=onehot_encoder(df["density"])
    name_encoder=onehot_encoder(df["name of product"])

    #Join three variables
    def scaling(data):
        scaled_data=(data-data.min(axis=0))/(data.max(axis=0)-data.min(axis=0))
        return scaled_data
    scaled=scaling(df.iloc[:,[0,4,6]])
    ## transform data
    new_data=scaled.join(pd.DataFrame(density_encoder)).join(pd.DataFrame(name_encoder),lsuffix="left",rsuffix='_other')

    def convert_data(data,scaled_data):
        data=scaled_data*(data.max(axis=0)-data.min(axis=0))+data.min(axis=0)
        return data

    ######################Splitting the data into training and testing parts
    y_names=["total processing time",'machine numbers','name of product',"machine"]
    x_names=["total processing time",'machine numbers']
    x=new_data.iloc[:,~new_data.columns.isin(x_names)]
    y=df[y_names]
    #For modeling easily
    y["machine numbers"].replace({1:0,2:1,3:2,4:3,5:4,6:5,8:6},inplace=True)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    def MAPE(Y_actual,Y_Predicted):
        mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
        return mape


    def Create_Model(trainx, trainy, testx, testy, save_path):
        alpha = 0.5
        input_shape = (200, 1)
        classes = len(np.unique(trainy.iloc[:, 1]))

        input_layer = Input(input_shape)
        conv1 = Conv1D(32, 64, padding="valid", activation="relu")(input_layer)
        max1 = MaxPool1D(6)(conv1)

        conv2 = Conv1D(64, 16, padding="same", activation="relu")(max1)
        max2 = MaxPool1D(3)(conv2)

        conv3 = Conv1D(128, 8, padding="same", activation="relu")(max2)
        max3 = MaxPool1D(3)(conv3)

        flatten = Flatten()(max3)

        out1 = Dense(classes, activation="softmax", name="machine_numbers")(flatten)
        out2 = Dense(1, activation="linear", name="process_time")(flatten)

        model = keras.models.Model(inputs=[input_layer], outputs=[out1, out2])

        model.compile(loss={'machine_numbers': 'categorical_crossentropy', 'process_time': 'mae'}, optimizer="adam",
                      loss_weights={'machine_numbers': 1 - alpha, 'process_time': alpha})

        path = "./model/%s.hdf5" % save_path
        keras_callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, mode='min', min_delta=0.000001),
            ModelCheckpoint(path, monitor='val_loss', save_best_only=True, mode='min')]

        # fit the keras model on the dataset
        his = model.fit(trainx.values.reshape(len(trainx), input_shape[0], 1),
                        [to_categorical(trainy.iloc[:, 1]), trainy.iloc[:, 0]],
                        epochs=100,
                        batch_size=100,
                        validation_split=0.2,
                        callbacks=keras_callbacks)

        pre = model.predict(testx.values.reshape(len(testx), 200, 1))
        acc = sum((testy.iloc[:, 1]) == pre[0].argmax(axis=1)) / float(len(testy))
        mape_times = MAPE(np.array(testy.iloc[:, 0]), np.array(pre[1]))

        return his, model, pre, acc, mape_times

    his,model,pre,acc,mape_times=Create_Model(x_train,y_train.iloc[:,[0,1]],x_test,y_test.iloc[:,[0,1]],"cnn")
    y_test["machine numbers"].replace({0:1,1:2,2:3,3:4,4:5,5:6,6:8},inplace=True)
    pre_df=pd.DataFrame(pre[1].round(),columns=["total processing time"])
    pre_df["machine numbers"]=pd.DataFrame(pre[0].argmax(axis=1))
    pre_df["machine numbers"].replace({0:1,1:2,2:3,3:4,4:5,5:6,6:8},inplace=True)
    pre_df["machine ID"]=y.iloc[pre_df.index]["machine"]
    pre_df["name of product"]=y.iloc[pre_df.index]["name of product"]
    job_numbers=len(pre_df.iloc[0:150,].groupby(["name of product","machine ID"]).sum().index.levels[0])
    machines_numbers=len(pre_df.iloc[0:150,].groupby(["name of product","machine ID"]).sum().index.levels[1])
    processing_time=pre_df.iloc[0:150,].groupby(["name of product","machine ID"]).sum()["total processing time"]
    pre_df.iloc[0:150,].groupby(["name of product","machine ID"]).sum()


    def input_construction(pre_df):
        """df is one dataframe consists of processing time and corresponding job order"""

        df = pre_df.groupby(["name of product", "machine ID"]).sum()["total processing time"].unstack().fillna(0)
        job_numbers = df.shape[0]
        machines_numbers = max(df.columns)

        r = pre_df.groupby(["name of product", "machine ID"]).sum().index.levels[1].values

        R = np.zeros((job_numbers, machines_numbers))
        for i in range(job_numbers):
            R[i] = np.array(range(0, machines_numbers)) + 1

        P = np.zeros((job_numbers, machines_numbers))
        for i in range(job_numbers):
            P[i][r - 1] = df.iloc[i, :]

        return R.astype(int), P.astype(int)


    R, P = input_construction(pre_df.iloc[0:100, ])



    class Problem:
        m = None  # number of the machines
        n = None  # number of the jobs
        solute = None
        time_low = None
        time_high = None

        p = np.array([])  # the processing time of jobs
        r = np.array([])  # the order limit
        x = np.array([])  # the result position mat
        h = np.array([])  # the start time of jobs
        e = np.array([])  # the end time of jobs
        f = np.array([])

        start = np.array([])  ############Start time for each action
        end = np.array([])  ############End time for eaCH ACTION
        best_x = np.array([])
        optimal_x = None
        number_of_1d_feature = 18

        def __init__(self, m, n, time_low, time_high, bool_random):
            self.m = m
            self.n = n
            self.time_low = time_low
            self.time_high = time_high
            self.solute = 0
            self.bool_generate_random_JSSP = bool_random
            self.GenerateRandomProblem()

        def GenerateRandomProblem(self):
            n = self.n
            m = self.m
            if self.bool_generate_random_JSSP == True:
                a = list(range(self.time_low, self.time_high))
                p = []
                for k in range(self.n):
                    p.append(random.sample(a, self.m))
                self.p = np.array(p)

                a = list(range(self.m))
                r = []
                for k in range(self.n):
                    r.append(random.sample(a, self.m))
                self.r = np.array(r)

                sum_time_of_job = np.sum(self.p, axis=1)

                for i in range(n):
                    for j in range(i + 1, n):
                        if sum_time_of_job[i] > sum_time_of_job[j]:
                            a = np.copy(self.p[j, :])
                            self.p[j, :] = self.p[i, :]
                            self.p[i, :] = a
                            sum_time_of_job[i], sum_time_of_job[j] = sum_time_of_job[j], sum_time_of_job[i]

                sum_time_of_mach = [[i, 0] for i in range(m)]
                for i in range(n):
                    for j in range(m):
                        sum_time_of_mach[self.r[i, j]][1] += self.p[i, j]

                for i in range(m):
                    for j in range(i + 1, m):
                        if sum_time_of_mach[i][1] > sum_time_of_mach[j][1]:
                            sum_time_of_mach[i], sum_time_of_mach[j] = sum_time_of_mach[j], sum_time_of_mach[i]

                nr = np.zeros((n, m), dtype=int) - 1
                for i in range(m):
                    nr[self.r == i] = sum_time_of_mach[i][0]

                sum_time_of_mach = [[i, 0] for i in range(m)]
                for i in range(n):
                    for j in range(m):
                        sum_time_of_mach[self.r[i, j]][1] += self.p[i, j]

                self.r = nr
            else:
                self.r = R - 1
                self.p = P

        def Print_info(self):

            machine_job_p = np.zeros((self.m, self.n))
            machine_job_r = np.zeros((self.m, self.n))

            for job in range(self.n):
                for order in range(self.m):
                    machine = self.r[job, order]
                    machine_job_p[machine, job] = self.p[job, order]
                    machine_job_r[machine, job] = order

            np.savetxt('p.csv', machine_job_p, delimiter=',')
            np.savetxt('r.csv', machine_job_r, delimiter=',')
            self.PlotResult()
            return machine_job_p, machine_job_r

        def SaveProblemToFile(self, filepath, index, pool=0):

            filename = '{}/jssp_problem_m={}_n={}_timehigh={}_timelow={}_pool={}.txt'.format(
                filepath, self.m, self.n, self.time_high, self.time_low, pool)
            f = open(filename, 'a')
            # f.write(str(index))
            # f.write('\nr=\n')
            f.write(str(self.m) + '\n')
            f.write(str(self.n) + '\n')
            f.write(TranslateNpToStr(self.p))
            f.write(TranslateNpToStr(self.r))
            f.close()

        def SavesolutionToFile(self, filepath, index, pool=0):
            f = open('{}/jssp_problem_m={}_n={}_timehigh={}_timelow={}_pool={}.txt'.format(
                filepath, self.m, self.n, self.time_high, self.time_low, pool), 'a')
            # f.write(str(index)+'\n')
            f.write(TranslateNpToStr(self.x))
            f.write(TranslateNpToStr(self.h))
            f.write(TranslateNpToStr(self.e))
            f.close()

        def SoluteWithBBM(self):
            solver = pywrapcp.Solver('jobshop')
            solver.TimeLimit(1)

            all_machines = range(0, self.m)
            all_jobs = range(0, self.n)

            x = np.zeros((self.m, self.n, 2), dtype=int)
            h = np.zeros((self.m, self.n), dtype=int)
            e = np.zeros((self.m, self.n), dtype=int)

            horizon = int(self.p.sum())
            # Creates jobs.
            all_tasks = {}
            for i in all_jobs:
                for j in range(self.m):
                    all_tasks[(i, j)] = solver.FixedDurationIntervalVar(
                        0, horizon, int(self.p[i, j]), False, 'Job_%i_%i' % (i, j))

            # Creates sequence variables and add disjunctive constraints.
            all_sequences = []
            all_machines_jobs = []
            for i in all_machines:

                machines_jobs = []
                for j in all_jobs:
                    for k in range(self.m):
                        if self.r[j, k] == i:
                            machines_jobs.append(all_tasks[(j, k)])
                disj = solver.DisjunctiveConstraint(
                    machines_jobs, 'machine %i' % i)
                all_sequences.append(disj.SequenceVar())
                solver.Add(disj)

            # Add conjunctive contraints.
            for i in all_jobs:
                for j in range(0, self.m - 1):
                    solver.Add(
                        all_tasks[(i, j + 1)].StartsAfterEnd(all_tasks[(i, j)]))

            # Set the objective.
            obj_var = solver.Max([all_tasks[(i, self.m - 1)].EndExpr()
                                  for i in all_jobs])
            objective_monitor = solver.Minimize(obj_var, 1)
            # Create search phases.
            sequence_phase = solver.Phase([all_sequences[i] for i in all_machines],
                                          solver.SEQUENCE_DEFAULT)
            vars_phase = solver.Phase([obj_var],
                                      solver.CHOOSE_FIRST_UNBOUND,
                                      solver.ASSIGN_MIN_VALUE)
            main_phase = solver.Compose([sequence_phase, vars_phase])
            # Create the solution collector.
            collector = solver.LastSolutionCollector()

            # Add the interesting variables to the SolutionCollector.
            collector.Add(all_sequences)
            collector.AddObjective(obj_var)

            for i in all_machines:
                sequence = all_sequences[i]
                sequence_count = sequence.Size()
                for j in range(0, sequence_count):
                    t = sequence.Interval(j)
                    collector.Add(t.StartExpr().Var())
                    collector.Add(t.EndExpr().Var())
            # Solve the problem.
            disp_col_width = 10
            if solver.Solve(main_phase, [objective_monitor, collector]):
                # print("\nOptimal Schedule Length:", collector.ObjectiveValue(0), "\n")
                sol_line = ""
                sol_line_tasks = ""
                # print("Optimal Schedule", "\n")

                for i in all_machines:
                    seq = all_sequences[i]
                    sol_line += "Machine " + str(i) + ": "
                    sol_line_tasks += "Machine " + str(i) + ": "
                    sequence = collector.ForwardSequence(0, seq)
                    seq_size = len(sequence)

                    for j in range(0, seq_size):
                        t = seq.Interval(sequence[j])
                        # Add spaces to output to align columns.
                        sol_line_tasks += t.Name() + " " * (disp_col_width - len(t.Name()))
                        x[i, j, 0] = int(t.Name().split('_')[1])
                        x[i, j, 1] = int(t.Name().split('_')[2])

                    for j in range(0, seq_size):
                        t = seq.Interval(sequence[j])
                        sol_tmp = "[" + str(collector.Value(0, t.StartExpr().Var())) + ","
                        sol_tmp += str(collector.Value(0,
                                                       t.EndExpr().Var())) + "] "
                        # Add spaces to output to align columns.
                        sol_line += sol_tmp + " " * (disp_col_width - len(sol_tmp))

                        h[i, j] = collector.Value(0, t.StartExpr().Var())
                        e[i, j] = collector.Value(0, t.EndExpr().Var())
                        self.start = h
                        self.end = e

                    sol_line += "\n"
                    sol_line_tasks += "\n"

            self.x = x
            self.h = h
            self.e = e
            self.best_x = x

        def SoluteWithGA(self):

            pt_tmp = self.p
            ms_tmp = self.r + 1

            dfshape = pt_tmp.shape
            print(dfshape)
            num_mc = dfshape[1]  # number of machines
            num_job = dfshape[0]  # number of jobs
            num_gene = num_mc * num_job  # number of genes in a chromosome

            pt = pt_tmp
            ms = ms_tmp

            population_size = 30
            crossover_rate = 0.8
            mutation_rate = 0.2
            mutation_selection_rate = 0.2
            num_mutation_jobs = int(round(num_gene * mutation_selection_rate))
            num_iteration = 2000

            start_time = time.time()

            Tbest = 999999999999999
            best_list, best_obj = [], []
            population_list = []
            makespan_record = []
            for i in range(population_size):
                # generate a random permutation of 0 to num_job*num_mc-1
                nxm_random_num = list(np.random.permutation(num_gene))
                # add to the population_list
                population_list.append(nxm_random_num)
                for j in range(num_gene):
                    # convert to job number format, every job appears m times
                    population_list[i][j] = population_list[i][j] % num_job

            for n in range(num_iteration):
                Tbest_now = 99999999999

                '''-------- two point crossover --------'''
                parent_list = copy.deepcopy(population_list)
                offspring_list = copy.deepcopy(population_list)
                # generate a random sequence to select the parent chromosome to crossover
                S = list(np.random.permutation(population_size))

                for m in range(int(population_size / 2)):
                    crossover_prob = np.random.rand()
                    if crossover_rate >= crossover_prob:
                        parent_1 = population_list[S[2 * m]][:]
                        parent_2 = population_list[S[2 * m + 1]][:]
                        child_1 = parent_1[:]
                        child_2 = parent_2[:]
                        cutpoint = list(np.random.choice(
                            num_gene, 2, replace=False))
                        cutpoint.sort()

                        child_1[cutpoint[0]:cutpoint[1]
                        ] = parent_2[cutpoint[0]:cutpoint[1]]
                        child_2[cutpoint[0]:cutpoint[1]
                        ] = parent_1[cutpoint[0]:cutpoint[1]]
                        offspring_list[S[2 * m]] = child_1[:]
                        offspring_list[S[2 * m + 1]] = child_2[:]

                '''----------repairment-------------'''
                for m in range(population_size):
                    job_count = {}
                    # 'larger' record jobs appear in the chromosome more than m times, and 'less' records less than m times.
                    larger, less = [], []
                    for i in range(num_job):
                        if i in offspring_list[m]:
                            count = offspring_list[m].count(i)
                            pos = offspring_list[m].index(i)
                            # store the above two values to the job_count dictionary
                            job_count[i] = [count, pos]
                        else:
                            count = 0
                            job_count[i] = [count, 0]
                        if count > num_mc:
                            larger.append(i)
                        elif count < num_mc:
                            less.append(i)

                    for k in range(len(larger)):
                        chg_job = larger[k]
                        while job_count[chg_job][0] > num_mc:
                            for d in range(len(less)):
                                if job_count[less[d]][0] < num_mc:
                                    offspring_list[m][job_count[chg_job]
                                    [1]] = less[d]
                                    job_count[chg_job][1] = offspring_list[m].index(
                                        chg_job)
                                    job_count[chg_job][0] = job_count[chg_job][0] - 1
                                    job_count[less[d]][0] = job_count[less[d]][0] + 1
                                if job_count[chg_job][0] == num_mc:
                                    break

                '''--------mutatuon--------'''
                for m in range(len(offspring_list)):
                    mutation_prob = np.random.rand()
                    if mutation_rate >= mutation_prob:
                        # chooses the position to mutation
                        m_chg = list(np.random.choice(
                            num_gene, num_mutation_jobs, replace=False))
                        # save the value which is on the first mutation position
                        t_value_last = offspring_list[m][m_chg[0]]
                        for i in range(num_mutation_jobs - 1):
                            # displacement
                            offspring_list[m][m_chg[i]
                            ] = offspring_list[m][m_chg[i + 1]]

                        # move the value of the first mutation position to the last mutation position
                        offspring_list[m][m_chg[num_mutation_jobs - 1]
                        ] = t_value_last

                '''--------fitness value(calculate makespan)-------------'''
                total_chromosome = copy.deepcopy(
                    parent_list) + copy.deepcopy(offspring_list)  # parent and offspring chromosomes combination
                chrom_fitness, chrom_fit = [], []
                total_fitness = 0
                for m in range(population_size * 2):  # for every gene line
                    j_keys = [j for j in range(num_job)]
                    key_count = {key: 0 for key in j_keys}
                    j_count = {key: 0 for key in j_keys}
                    m_keys = [j + 1 for j in range(num_mc)]
                    m_count = {key: 0 for key in m_keys}

                    for i in total_chromosome[m]:
                        gen_t = int(pt[i][key_count[i]])
                        gen_m = int(ms[i][key_count[i]])
                        j_count[i] = j_count[i] + gen_t
                        m_count[gen_m] = m_count[gen_m] + gen_t

                        if m_count[gen_m] < j_count[i]:
                            m_count[gen_m] = j_count[i]
                        elif m_count[gen_m] > j_count[i]:
                            j_count[i] = m_count[gen_m]

                        key_count[i] = key_count[i] + 1

                    makespan = max(j_count.values())
                    chrom_fitness.append(1 / makespan)
                    chrom_fit.append(makespan)
                    total_fitness = total_fitness + chrom_fitness[m] + 0.01

                '''----------selection(roulette wheel approach)----------'''
                pk, qk = [], []

                for i in range(population_size * 2):
                    pk.append(chrom_fitness[i] / total_fitness)
                for i in range(population_size * 2):
                    cumulative = 0
                    for j in range(0, i + 1):
                        cumulative = cumulative + pk[j]
                    qk.append(cumulative)

                selection_rand = [np.random.rand() for i in range(population_size)]

                for i in range(population_size):
                    if selection_rand[i] <= qk[0]:
                        population_list[i] = copy.deepcopy(total_chromosome[0])
                    else:
                        for j in range(0, population_size * 2 - 1):
                            if selection_rand[i] > qk[j] and selection_rand[i] <= qk[j + 1]:
                                population_list[i] = copy.deepcopy(
                                    total_chromosome[j + 1])
                                break
                '''----------comparison----------'''
                for i in range(population_size * 2):
                    if chrom_fit[i] < Tbest_now:
                        Tbest_now = chrom_fit[i]
                        sequence_now = copy.deepcopy(total_chromosome[i])
                if Tbest_now <= Tbest:
                    Tbest = Tbest_now
                    sequence_best = copy.deepcopy(sequence_now)

                makespan_record.append(Tbest)

            x = np.zeros((self.m, self.n, 2), dtype=int)
            h = np.zeros((self.m, self.n), dtype=int)
            e = np.zeros((self.m, self.n), dtype=int)

            m_keys = [j + 1 for j in range(num_mc)]
            j_keys = [j for j in range(num_job)]

            key_count = {key: 0 for key in j_keys}
            j_count = {key: 0 for key in j_keys}
            m_count = {key: 0 for key in m_keys}
            j_record = {}
            for i in sequence_best:
                gen_t = int(pt[i][key_count[i]])  # time
                gen_m = int(ms[i][key_count[i]])  # order
                j_count[i] = j_count[i] + gen_t  # time of job
                m_count[gen_m] = m_count[gen_m] + gen_t  # time of machine

                if m_count[gen_m] < j_count[i]:
                    m_count[gen_m] = j_count[i]
                elif m_count[gen_m] > j_count[i]:
                    j_count[i] = m_count[gen_m]

                # convert seconds to hours, minutes and seconds
                start_time = int(j_count[i] - pt[i][int(key_count[i])])
                end_time = int(j_count[i])

                j_record[(i, gen_m)] = [start_time, end_time, key_count[i]]

                key_count[i] = key_count[i] + 1

            df = []

            for m in m_keys:
                for j in j_keys:
                    list_of_start = [j_record[(q, m)][0] for q in j_keys]
                    list_of_start.sort()
                    order = list_of_start.index(j_record[(j, m)][0])
                    h[m - 1, order] = j_record[(j, m)][0]
                    e[m - 1, order] = j_record[(j, m)][1]
                    x[m - 1, order, 0] = j
                    x[m - 1, order, 1] = j_record[(j, m)][2]
                    df.append(dict(Task='Machine %s' % (m), Start='2018-07-14 %s' % (str(j_record[(
                        j, m)][0])), Finish='2018-07-14 %s' % (str(j_record[(j, m)][1])), Resource='Job %s' % (j + 1)))
                    self.start = h
                    self.end = e
            self.h = h
            self.e = e
            self.x = x
            self.best_x = x

        def PlotResult(self, num=0):

            colorbox = ['yellow', 'whitesmoke', 'lightyellow',
                        'khaki', 'silver', 'pink', 'lightgreen', 'orange', 'grey', '#8ca8df', 'brown']
            result = {}
            for i in range(100):
                colorArr = ['1', '2', '3', '4', '5', '6', '7',
                            '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
                color = ""
                for i in range(6):
                    color += colorArr[random.randint(0, 14)]
                colorbox.append("#" + color)

            zzl = plt.figure(figsize=(12, 4))
            for i in range(self.m):
                # number_of_mashine:
                for j in range(self.n):
                    # number_of_job:

                    mPoint1 = self.h[i, j]
                    mPoint2 = self.e[i, j]
                    mText = i + 1.5
                    PlotRec(mPoint1, mPoint2, mText)
                    Word = str(self.x[i, j, 0] + 1) + '.' + str(self.x[i, j, 1] + 1)
                    # hold on

                    x1 = mPoint1
                    y1 = mText - 0.8
                    x2 = mPoint2
                    y2 = mText - 0.8
                    x3 = mPoint2
                    y3 = mText
                    x4 = mPoint1
                    y4 = mText

                    plt.fill([x1, x2, x3, x4], [y1, y2, y3, y4],
                             color=colorbox[self.x[i, j, 0]])

                    plt.text(0.5 * mPoint1 + 0.5 * mPoint2 - 3, mText - 0.5, Word)
            plt.xlabel('Time')
            plt.ylabel('Machine')
            plt.tight_layout()
            # plt.savefig('../results/out.png', dpi=700)

            return result

        def SoluteWithBBMAndSaveToFile(self, filepath, index, pool=0):
            self.SoluteWithBBM()
            self.SaveProblemToFile(filepath, index, pool)
            self.SavesolutionToFile(filepath, index, pool)

        def SoluteWithGaAndSaveToFile(self, filepath, index, pool=0):
            self.SoluteWithGA()
            self.SaveProblemToFile(filepath, index, pool)
            self.SavesolutionToFile(filepath, index, pool)

        def LoadProblemWithSolution(self, filepath, index, pool=0):
            f = open('data/jssp_problem_' + filepath, 'r')
            # line_list = [ i for i in range(index * 7, index * 7 + 7 )]
            data = f.readlines()
            data = data[index * 7:index * 7 + 7]
            self.m = int(data[0])
            self.n = int(data[1])
            # print(data[2])
            self.p = np.fromstring(
                data[2][:-1], dtype=int, sep=',').reshape((self.n, self.m))
            self.r = np.fromstring(
                data[3][:-1], dtype=int, sep=',').reshape((self.n, self.m))
            self.x = np.fromstring(
                data[4][:-1], dtype=int, sep=',').reshape((self.m, self.n, 2))
            self.h = np.fromstring(
                data[5][:-1], dtype=int, sep=',').reshape((self.m, self.n))
            self.e = np.fromstring(
                data[6][:-1], dtype=int, sep=',').reshape((self.m, self.n))

        def CalculateSimilarityDegree(self, x=None):

            if x is not None:
                self.x = x

            right = 0
            for i in range(self.m):
                for j in range(self.n):
                    if self.x[i, j, 0] == self.best_x[i, j, 0] and self.x[i, j, 1] == self.best_x[i, j, 1]:
                        right += 1

            return right / self.m / self.n

        def LoadProblemWithoutSolution(self, filepath, index, pool=0):
            f = open('data/jssp_problem_' + filepath, 'r')
            # line_list = [ i for i in range(index * 7, index * 7 + 7 )]
            data = f.readlines()
            data = data[index * 7:index * 7 + 7]
            self.m = int(data[0])
            self.n = int(data[1])
            # print(data[2])
            self.p = np.fromstring(
                data[2][:-1], dtype=int, sep=',').reshape((self.n, self.m))
            self.r = np.fromstring(
                data[3][:-1], dtype=int, sep=',').reshape((self.n, self.m))

        def subproblem(self):
            sub_list = []
            for job in range(self.n):
                for procedure in range(self.m):
                    sub_p = Subproblem(self, job, procedure)
                    sub_list.append(sub_p)
            return sub_list

        def GetFeaturesInTest1D(self):

            F = []
            sub_list = self.subproblem()
            for sub in sub_list:
                F.append(sub.GetFeatures1D())
            F = np.array(F)

            return F

        def Getlables(self):

            L = []
            sub_list = self.subproblem()
            for sub in sub_list:
                L.append(sub.label)
            L = np.array(L)

            return L

        def GetFeaturesInTest1D2D(self):
            len_feature_1d = self.number_of_1d_feature
            #         len_feature_detailed=self.n*self.m

            F1d = []
            # #         F_s=[]   ###########add
            # #         F_e=[]   ###########add

            #         F2d1 = []
            #         F2d2 = []
            #         F2d3 = []
            #         F2d4 = []
            #         F2d5 = []
            #         F2d6 = []

            sub_list = self.subproblem()
            for sub in sub_list:
                F1d.append(sub.GetFeatures1D())
            # #             F_s.append(sub.GetFeaturesDetailed()[0])
            # #             F_e.append(sub.GetFeaturesDetailed()[1])

            #             features_2d = sub.GetFeatures2D()
            #             F2d1.append(features_2d[0])
            #             F2d2.append(features_2d[1])
            #             F2d3.append(features_2d[2])
            #             F2d4.append(features_2d[3])
            #             F2d5.append(features_2d[4])
            #             F2d6.append(features_2d[5])

            F1d = np.array(F1d).reshape((-1, len_feature_1d, 1))
            # #         F_s=np.array(F_s).reshape((-1,len_feature_detailed,1))
            # #         F_e=np.array(F_e).reshape((-1,len_feature_detailed,1))

            #         F2d1 = np.array(F2d1).reshape(
            #             (-1, self.n ** 2, self.n ** 2, 1))
            #         F2d2 = np.array(F2d2).reshape(
            #             (-1, self.n ** 2, self.n, 1))
            #         F2d3 = np.array(F2d3).reshape(
            #             (-1, self.n ** 2, len_feature_1d, 1))
            #         F2d4 = np.array(F2d4).reshape(
            #             (-1, self.n, self.n, 1))
            #         F2d5 = np.array(F2d5).reshape(
            #             (-1, self.n, len_feature_1d, 1))
            #         F2d6 = np.array(F2d6).reshape(
            #             (-1, len_feature_1d, len_feature_1d, 1))

            #         tf = F1d.reshape((-1, len_feature_1d, 1))
            #               F_s.reshape((-1,len_feature_detailed,1)),
            #              F_e.reshape((-1,len_feature_detailed,1))]

            return F1d

        def GetIndexMatrix(self):
            index = np.zeros((self.n, self.m))

            sub = self.subproblem()
            for s in sub:
                index[s.job_id, s.machine_id] = s.SearchInX()
            return index

        def SchedulingSequenceGenerationMethod(self, output):
            np.savetxt('output.csv', output, fmt="%.2f", delimiter=',')
            for i in range(self.m * self.n):
                output[i, :] = output[i, :] / output[i, :].sum()

            h = np.zeros((self.m, self.n))
            e = np.zeros((self.m, self.n))
            x = np.zeros((self.m, self.n, 2))

            procedure_job = [0] * self.n
            order_machine = [0] * self.m

            for i in range(self.m * self.n):
                possible_probability = []

                for job in range(self.n):
                    procedure = procedure_job[job]
                    machine = self.r[job, min(4, procedure)]
                    order = order_machine[machine]
                    if procedure < 5 and order < 5:
                        possible_probability.append(
                            [job, procedure, machine, order, output[job * self.m + procedure][order]])
                    else:
                        machine = -1
                        possible_probability.append([job, procedure, machine, order, 0])

                possible_probability = sorted(possible_probability, key=lambda x: x[4])

                bestjob, bestproce, bestmachine, bestorder = possible_probability[-1][:4]
                x[bestmachine, bestorder][:] = [bestjob, bestproce]

                procedure_job[bestjob] += 1
                order_machine[bestmachine] += 1

            self.x = x

        def GurobiModelingmethod(self, output):
            np.savetxt('output.csv', output, fmt="%.2f", delimiter=',')
            lables = self.Getlables()
            R = self.r.reshape(self.m * self.n)
            # x ,p = guchoose.main(output,R,lables,self.m,self.n)

            x = [[] for j in range(self.m)]  ######################################################Chaning
            h = np.zeros((self.m, self.n))
            e = np.zeros((self.m, self.n))

            for order in range(self.n):
                timeline_machine = np.zeros((self.m), dtype=int)
                timeline_jobs = np.zeros((self.n), dtype=int)
                index_in_machine = np.zeros((self.m), dtype=int)
                job_finsh = np.zeros((self.n), dtype=int)
                for i in range(self.m * self.n):
                    mask = np.zeros((self.m), dtype=int)
                    for ma in range(self.m):
                        job, order = x[ma, min(self.n - 1, index_in_machine[ma]), :]
                        mask[ma] = timeline_machine[ma]

                    earlyestmachine = np.argmin(mask)

                    while index_in_machine[earlyestmachine] == self.n:
                        timeline_machine[earlyestmachine] = 100000
                        earlyestmachine = np.argmin(timeline_machine)
                    # while can_do_in_machine[earlyestmachine]

                    job, order = x[earlyestmachine,
                                 index_in_machine[earlyestmachine], :]
                    time_s = max(
                        timeline_machine[earlyestmachine], timeline_jobs[job])
                    time_e = time_s + self.p[job, order]
                    timeline_machine[earlyestmachine] = time_e
                    timeline_jobs[job] = time_e
                    h[earlyestmachine, index_in_machine[earlyestmachine]] = time_s
                    e[earlyestmachine, index_in_machine[earlyestmachine]] = time_e
                    index_in_machine[earlyestmachine] += 1
                    job_finsh[job] += 1

            self.e = e
            self.h = h

        def PriorityQueuingMethod(self, output):
            np.savetxt('output.csv', output, fmt="%.2f", delimiter=',')
            lables = self.Getlables()
            R = self.r.reshape(self.m * self.n)

            x = [[] for j in range(self.m)]
            for i in range(self.m * self.n):
                machine = R[i]
                x[machine].append([i // self.m, i % self.m, output[i]])

            for m in range(self.m):
                x[m].sort(key=lambda x: x[2])

            xx = np.zeros((self.m, self.n, 2), dtype=int)
            for i in range(self.m):
                for j in range(self.n):
                    xx[i, j, 0] = x[i][j][0]
                    xx[i, j, 1] = x[i][j][1]
            x = xx
            #         self.x = xx                         ####################################Changing
            h = np.zeros((self.m, self.n))
            e = np.zeros((self.m, self.n))

            for order in range(self.n):
                timeline_machine = np.zeros((self.m), dtype=int)
                timeline_jobs = np.zeros((self.n), dtype=int)
                index_in_machine = np.zeros((self.m), dtype=int)
                job_finsh = np.zeros((self.n), dtype=int)
                for i in range(self.m * self.n):
                    mask = np.zeros((self.m), dtype=int)
                    for ma in range(self.m):
                        job, order = x[ma, min(self.n - 1, index_in_machine[ma]), :]
                        if job_finsh[job] == order:
                            mask[ma] = timeline_machine[ma]
                        else:
                            mask[ma] = 10000
                    earlyestmachine = np.argmin(mask)

                    while index_in_machine[earlyestmachine] == self.n:
                        timeline_machine[earlyestmachine] = 100000
                        earlyestmachine = np.argmin(timeline_machine)
                    # while can_do_in_machine[earlyestmachine]

                    job, order = x[earlyestmachine,
                                 index_in_machine[earlyestmachine], :]
                    time_s = max(
                        timeline_machine[earlyestmachine], timeline_jobs[job])
                    time_e = time_s + self.p[job, order]
                    timeline_machine[earlyestmachine] = time_e
                    timeline_jobs[job] = time_e
                    h[earlyestmachine, index_in_machine[earlyestmachine]] = time_s
                    e[earlyestmachine, index_in_machine[earlyestmachine]] = time_e
                    index_in_machine[earlyestmachine] += 1
                    job_finsh[job] += 1

        def GetMakespan(self):
            return self.e.max()


    def TranslateNpToStr(m):
        a = m.reshape((-1))
        a = list(a)
        s = ''.join(['{},'.format(round(o, 2)) for o in a]) + '\n'
        return s


    def PlotRec(mPoint1, mPoint2, mText):
        vPoint = np.zeros((4, 2))
        vPoint[0, :] = [mPoint1, mText - 0.8]
        vPoint[1, :] = [mPoint2, mText - 0.8]
        vPoint[2, :] = [mPoint1, mText]
        vPoint[3, :] = [mPoint2, mText]
        plt.plot([vPoint[0, 0], vPoint[1, 0]], [vPoint[0, 1], vPoint[1, 1]], 'k')
        # hold on
        plt.plot([vPoint[0, 0], vPoint[2, 0]], [vPoint[0, 1], vPoint[2, 1]], 'k')
        plt.plot([vPoint[1, 0], vPoint[3, 0]], [vPoint[1, 1], vPoint[3, 1]], 'k')
        plt.plot([vPoint[2, 0], vPoint[3, 0]], [vPoint[2, 1], vPoint[3, 1]], 'k')


    class Subproblem:
        father_problem = None
        machine_id = None
        job_id = None
        procedure = None
        time = None
        num_of_machine = None
        num_of_job = None
        number_of_1D_feature = None
        features_1D = None
        features_2D = None
        order_in_machine = None

        label = []

        def __init__(self, fatherproblem, jobs, procedure):
            self.father_problem = fatherproblem
            self.job_id = jobs
            self.procedure = procedure
            self.machine_id = fatherproblem.r[jobs, procedure] - 1
            self.time = self.father_problem.p[self.job_id, self.machine_id]
            self.num_of_job = fatherproblem.p.shape[0]
            self.num_of_machine = fatherproblem.p.shape[1]

            self.label = self.SearchInX()
            self.time2 = self.label
            self.order_in_machine = self.SearchInX()
            self.features_1D = self.GetFeatures1D()
            self.features_2D = self.GetFeatures2D()

        def SearchInX(self):

            for i in range(self.num_of_machine):
                for j in range(self.num_of_job):
                    if self.job_id == self.father_problem.x[i, j, 0] and self.procedure == self.father_problem.x[i, j, 1]:
                        self.order_in_machine = j
                        return j
            return 'error'

        def CheckOrderInMachine(self):
            father_problem = self.father_problem
            joblist_in_machine = []
            for job in range(father_problem.n):
                for pro in range(father_problem.m):
                    machine = father_problem.r[job, pro]
                    if machine - 1 == self.machine_id:
                        joblist_in_machine.append([job, pro])

            joblist_in_machine.sort(key=lambda d: d[1])
            index = joblist_in_machine.index([self.job_id, self.procedure])
            return index

        def GetFeatures1D(self):
            # p [job,order]
            # r [job,order]

            features = np.zeros((18))

            start_time = np.zeros((self.father_problem.m, self.father_problem.n))
            start_end = np.zeros((self.father_problem.m, self.father_problem.n))

            T_max = float(self.father_problem.p.max())  #####################add
            T_min = float(self.father_problem.p.min()) + 0.0001  ####################add
            T_mean = float(self.father_problem.p.mean())  ####################add
            T_total = float(self.father_problem.p.sum())

            T_machine = float(self.father_problem.p[:, self.machine_id].sum()) + 0.0001
            T_job = float(self.father_problem.p[self.job_id, :].sum())
            T_item = float(self.time)
            order_of_procedure_in_machine = float(self.CheckOrderInMachine())

            ############Job
            features[0] = self.procedure / self.father_problem.n
            features[1] = T_item / T_total
            features[2] = T_item / T_max  ##################
            features[3] = T_item / T_min  ##################
            features[4] = T_item / T_mean  ##################

            features[5] = self.father_problem.p[self.job_id,
                          :self.procedure].sum() / T_job
            features[6] = self.father_problem.p[self.job_id, self.procedure] / self.father_problem.p[self.job_id,
                                                                               :self.procedure].sum()  #####################

            features[7] = self.father_problem.p[self.job_id,
                          self.procedure:].sum() / T_job

            features[8] = self.father_problem.p[self.job_id, self.procedure] / self.father_problem.p[self.job_id,
                                                                               self.procedure:].sum()  ######################

            features[9] = self.job_id / self.father_problem.n

            features[10] = T_item / T_machine
            features[11] = T_job / T_machine
            features[12] = np.sum(
                self.father_problem.r[:, self.procedure] == self.machine_id) / self.father_problem.n
            features[13] = order_of_procedure_in_machine / self.father_problem.n  #######Unused
            features[14] = self.machine_id / self.father_problem.m
            features[15] = T_item  #####################
            features[16] = order_of_procedure_in_machine  ############unused
            features[17] = self.procedure  ######################

            statr_time = self.father_problem.start
            end_time = self.father_problem.end

            self.number_of_1D_feature = features.shape[0]
            return features

        #     def GetFeaturesDetailed(self):
        #         features_start=self.father_problem.h
        #         features_end=self.father_problem.e
        #         return features_start,features_end

        def GetPLine(self):
            out = np.zeros((self.father_problem.n * self.father_problem.n, 1))
            sum_time_job = self.father_problem.p.sum(axis=1)
            for i in range(self.father_problem.n):
                for j in range(self.father_problem.n):
                    out[i * self.father_problem.n + j,
                        0] = sum_time_job[i] / sum_time_job[j]
            # assert(sum_time_job.shape == self.father_problem.n)
            return out

        def GetELine(self):
            out = np.zeros((self.father_problem.n, 1))
            sum_time_job = self.father_problem.p.sum(axis=1)
            for i in range(self.father_problem.n):
                out[i, 0] = self.time / sum_time_job[i]
            # assert(sum_time_job.shape == self.father_problem.n)
            return out

        def GetFeatures2D(self):
            features = []

            p_line = self.GetPLine()
            E_line = self.GetELine()
            f_line = self.features_1D.reshape((-1, 1))

            features.append(np.dot(p_line, p_line.T))
            features.append(np.dot(p_line, E_line.T))
            features.append(np.dot(p_line, f_line.T))
            features.append(np.dot(E_line, E_line.T))
            features.append(np.dot(E_line, f_line.T))
            features.append(np.dot(f_line, f_line.T))

            return features

        def Show2DFeatures(self):

            plt.figure(figsize=(6, 4))

            plt.subplot(231)
            plt.imshow(self.features_2D[0])
            plt.xlabel(r'$D^{2d}_{P^l,P^l}$')
            plt.subplot(232)
            plt.imshow(self.features_2D[1].T)
            plt.xlabel(r'$D^{2d}_{P^l,E^l}$')
            plt.subplot(233)
            plt.imshow(self.features_2D[2].T)
            plt.xlabel(r'$D^{2d}_{P^l,F^{l}_{ij}}$')
            plt.subplot(234)
            plt.imshow(self.features_2D[3])
            plt.xlabel(r'$D^{2d}_{E^l,E^l}$')
            plt.subplot(235)
            plt.imshow(self.features_2D[4])
            plt.xlabel(r'$D^{2d}_{E^l,F^{l}_{ij}}$')
            plt.subplot(236)
            plt.imshow(self.features_2D[5])
            plt.xlabel(r'$D^{2d}_{F^{l}_{ij},F^{l}_{ij}}$')
            plt.tight_layout()
            plt.savefig('figure/n={}m={}order={}.png'.format(self.machine_id,
                                                             self.job_id, self.procedure), dpi=500)

            plt.close()
            plt.show()

        def SaveFeaturesToFile(self, filepath, index, pool=0):

            father_problem = self.father_problem
            f = open('{}/jssp_feature_m={}_n={}_timehigh={}_timelow={}_pool={}.txt'.format(
                filepath, father_problem.m, father_problem.n, father_problem.time_high, father_problem.time_low, pool), 'a')
            f.write(TranslateNpToStr(self.features_1D))
            f.close()

        def SaveLablesToFile(self, filepath, index, pool=0):

            father_problem = self.father_problem
            probinfo = 'm={}_n={}_timehigh={}_timelow={}_pool={}.txt'.format(
                father_problem.m, father_problem.n, father_problem.time_high, father_problem.time_low, pool)

            f = open('{}/jssp_label_m={}_n={}_timehigh={}_timelow={}_pool={}.txt'.format(
                filepath, father_problem.m, father_problem.n, father_problem.time_high, father_problem.time_low, pool), 'a')
            f.write(str(self.label) + '\n')
            f.close()

            return probinfo

        def LoadFeatures(self, filepath, index, pool=0):

            father_problem = self.father_problem
            f = open('{}/jssp_problem_m={}_n={}_timehigh={}_timelow={}_pool={}.txt'.format(
                filepath, father_problem.m, father_problem.n, father_problem.time_high, father_problem.time_low, pool), 'r')
            data = f.readlines()
            data = data[index * 7:index * 7 + 7]

            len_of_1D_features = np.fromstring(
                data[0][:-1], dtype=float, sep=',').shape[0]
            len_of_m_muti_n = int(np.sqrt(np.fromstring(
                data[1][:-1], dtype=float, sep=',').shape))

            self.features_1D = np.fromstring(data[0][:-1], dtype=float, sep=',')
            featrue_2D = []
            featrue_2D.append(np.fromstring(
                data[1][:-1], dtype=float, sep=',').reshape((len_of_m_muti_n, len_of_m_muti_n)))
            featrue_2D.append(np.fromstring(
                data[2][:-1], dtype=float, sep=',').reshape((len_of_m_muti_n, len_of_m_muti_n)))
            featrue_2D.append(np.fromstring(
                data[3][:-1], dtype=float, sep=',').reshape((len_of_m_muti_n, len_of_1D_features)))
            featrue_2D.append(np.fromstring(
                data[4][:-1], dtype=float, sep=',').reshape((len_of_m_muti_n, len_of_m_muti_n)))
            featrue_2D.append(np.fromstring(
                data[5][:-1], dtype=float, sep=',').reshape((len_of_m_muti_n, len_of_1D_features)))
            featrue_2D.append(np.fromstring(
                data[6][:-1], dtype=float, sep=',').reshape((len_of_1D_features, len_of_1D_features)))
            f.close()

            self.features_2D = featrue_2D
            return self.features_1D, self.features_2D


    def CreateJssp_BBM(number_of_problem, index_cpu, m, n,
                       timehigh, timelow,
                       solution_path_bbm, feature_path_bbm, label_path_bbm,
                       solution_path_ga, feature_path_ga, label_path_ga,
                       bool_random):
        # Create Jobshop problem with ortools and save it to 'bigdata/'
        # problem can be solute with two main method:
        #   1. Branch and bound method (BBM) for small problems
        #   2. Genetic Method(GA) for big problems
        # Input:
        #   number_of_loop: is the number of problems need to create
        #   index_cpu: only used in muti cpu:
        #   m: the number of the machine in jobshop problem
        #   n: the number of the job in the job problem
        #   timehigh: the max producing time of the job's one processing
        #   timelow: the min producing time of the job's one processing

        pbar = progressbar.ProgressBar().start()
        for p in range(number_of_problem):

            pbar.update(int((p / (number_of_problem - 1)) * 100))

            # init one Jobshop problem randomly
            prob = Problem(m, n, time_low=timelow, time_high=timehigh, bool_random=bool_random)
            # solute the problem with two method, you can change it
            # if you do not the the ortools wheels, choose the GA method
            # prob.SoluteWithGaAndSaveToFile('bigdata/data', 0)
            prob.SoluteWithBBMAndSaveToFile(solution_path_bbm, 0)

            # print the information of the problem and the solution, and save it in to the file 'bigdata/'
            # prob.Print_info()
            sub_list = prob.subproblem()
            for i, subp in enumerate(sub_list):
                subp.SaveFeaturesToFile(feature_path_bbm, i)
                info = subp.SaveLablesToFile(label_path_bbm, i)

        pbar.finish()


    def loadFeaturesAndLabels(feature_path, label_path):
        features_1D_list = []
        label_list = []

        ###########Feature
        with open(feature_path, 'r') as f:
            while True:
                data = f.readline()
                if data != '':
                    features_1D = np.fromstring(data, dtype=float, sep=',')
                    features_1D_list.append(features_1D[0:18])
                else:
                    break

        ############label
        with open(label_path, 'r') as f:
            while True:
                data = f.readline()
                if data != '':
                    label_list.append(int(data))
                else:
                    break

        return np.asanyarray(features_1D_list), np.asanyarray(label_list)


    ################features have 11 elements


    # In[23]:


    from sklearn.cluster import KMeans


    # function returns WSS score for k values from 1 to kmax
    def calculate_WSS(points, kmax):
        sse = []
        for k in range(1, kmax + 1):
            kmeans = KMeans(n_clusters=k).fit(points)
            centroids = kmeans.cluster_centers_
            pred_clusters = kmeans.predict(points)
            curr_sse = 0
            # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
            for i in range(len(points)):
                curr_center = centroids[pred_clusters[i]]
                curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2
            sse.append(curr_sse)


    def normilization(data):
        """DATA NORMILIZATION
           The result willl be a list"""
        dataset = data
        max_value = np.max(dataset)
        min_value = np.min(dataset)
        scalar = max_value - min_value
        dataset = (dataset - min_value) / (scalar)
        return dataset


    def SS_lstm(trainX1, trainY1, testX1, testY1, trainX2, testX2, model_path):
        alpha = 0.4
        input_shape = (18, 1)
        classes = len(np.unique(trainY1))

        input2 = Input((1, 1))

        input1 = Input(input_shape)
        batch = BatchNormalization()(input1)
        lstm1 = LSTM(64, activation="relu", return_sequences=True)(batch)
        lstm2 = LSTM(128, activation="relu", return_sequences=False)(lstm1)

        flatten1 = Flatten()(input2)
        fetures = keras.layers.concatenate([lstm2, flatten1])

        out1 = Dense(classes, activation='softmax', name="classification")(fetures)

        repeat_vector = RepeatVector(input_shape[0])(lstm2)  ###############inpuu_shape[0] is time_steps

        decoder_lstm2 = LSTM(128, activation="relu", return_sequences=True)(repeat_vector)
        decoder_lstm1 = LSTM(64, activation="relu", return_sequences=True)(decoder_lstm2)

        out2 = Dense(input_shape[1], activation="linear", name="linear_regression")(decoder_lstm1)

        model = keras.models.Model(inputs=[input1, input2], outputs=[out1, out2])

        model.compile(loss={'classification': 'categorical_crossentropy', 'linear_regression': 'mse'}, optimizer="adam",
                      loss_weights={'classification': 1 - alpha, 'linear_regression': alpha})

        model.summary()

        path = model_path
        keras_callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, mode='min', min_delta=0.000001),
            ModelCheckpoint(path, monitor='val_loss', save_best_only=True, mode='min')]

        his = model.fit([trainX1.reshape(len(trainX1), input_shape[0], input_shape[1]),
                         trainX2.reshape(len(trainX2), 1, 1)],
                        [to_categorical(trainY1), trainX1.reshape(len(trainX1), 18, 1)],
                        validation_data=[[testX1.reshape(len(testX1), input_shape[0], input_shape[1]),
                                          testX2.reshape(len(testX2), 1, 1)],
                                         [to_categorical(testY1), testX1.reshape(len(testX1), 18, 1)]],
                        callbacks=keras_callbacks,
                        batch_size=24, epochs=150)
        return model, his


    # ## Testing
    ####### Generating the testing sampels using GA or BBM
    def generate_new_jssp_GA(m, n, time_low, time_high, bool_random):
        new_pro = Problem(m, n, time_low, time_high, bool_random=bool_random)
        new_pro.SoluteWithGA()
        best_mak = new_pro.GetMakespan()
        features = new_pro.GetFeaturesInTest1D2D()

        return features, best_mak, new_pro


    from keras.models import load_model


    # In[31]:


    def generating_features_labels(m, n, nubmers_of_problems):
        """ft10by10 JSSP training instances if m,n is setted as 10 by 10"""
        # Generating the training samples using optimal methods
        m = m
        n = n
        timehigh = 50
        timelow = 10
        pool = 0
        CreateJssp_BBM(nubmers_of_problems, pool, m, n, timehigh, timelow, filedir + "solution/BBM/ft10by10",
                       filedir + "feature/BBM/ft10by10",
                       filedir + "label/BBM/ft10by10",
                       filedir + "solution/GA/ft10by10",
                       filedir + "feature/GA/ft10by10",
                       filedir + "label/GA/ft10by10",
                       True)

        # Loading the generating samples including features and lables
        features_1D_bbm, labels_1D_bbm = loadFeaturesAndLabels(
            filedir + "feature/BBM/ft10by10/jssp_feature_m={}_n={}_timehigh=50_timelow=10_pool=0.txt".format(m, n),
            filedir + "label/BBM/ft10by10/jssp_label_m={}_n={}_timehigh=50_timelow=10_pool=0.txt".format(m, n))

        # Caculating the system features using the Kmeans algorithm
        WSS = calculate_WSS(features_1D_bbm, 30)
        kmeans = KMeans(n_clusters=12).fit(features_1D_bbm)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(normilization(features_1D_bbm))
        df = pd.DataFrame(normilization(features_1D_bbm), pred_clusters)
        df["type"] = pred_clusters
        features_1D_bbm = np.array(df)

        return features_1D_bbm, labels_1D_bbm


    def training_model(features, labels):
        X_train_bbm, X_test_bbm, y_train_bbm, y_test_bbm = train_test_split(features, labels, test_size=0.2,
                                                                            random_state=42)
        ss_lstm, his_sslstm = SS_lstm(X_train_bbm[:, 0:18], y_train_bbm, X_test_bbm[:, 0:18], y_test_bbm,
                                      X_train_bbm[:, -1], X_test_bbm[:, -1],
                                      "../model/fusion/best.hdf5")


    def solving_new_jssp(new_m, new_n):
        """new_m: new JSSP's m
           new_n: new JSSP's n
        """
        # JSSP instance
        features_ft10_10by10_bbm, mak_bbm_ft10_10by_10_ga, new_pro_ga = generate_new_jssp_GA(new_m, new_n, 100, 10, False)

        #     ###Step 1.1 Loading the model
        #     bestss_model_bbm=load_model("../model/fusion/best.hdf5")
        #     ###Step 1.2 Peparing the system-level features using K-means
        #     kmeans=KMeans(n_clusters =12).fit(normilization(features_ft10_10by10_bbm.reshape(features_ft10_10by10_bbm.shape[0],18)))
        #     centroids = kmeans.cluster_centers_
        #     pred_types = kmeans.predict(normilization(features_ft10_10by10_bbm.reshape(features_ft10_10by10_bbm.shape[0],18)))
        #     ###Step 1.3 Preparing the details-level feature
        #     features_ft10_10by10_bbm_types=pd.DataFrame(normilization(features_ft10_10by10_bbm.reshape(features_ft10_10by10_bbm.shape[0],18)))
        #     features_ft10_10by10_bbm_types["type"]=pred_types

        #     sslstm_bbm=bestss_model_bbm.predict([normilization(np.asarray(features_ft10_10by10_bbm_types)[:,0:18].reshape(new_m*new_n,18,1)),
        #                                                    np.asarray(features_ft10_10by10_bbm_types)[:,-1].reshape(new_m*new_n,1,1)])
        #     ###########lstm_bbm
        #     new_pro_ga.PriorityQueuingMethod(np.argmax(sslstm_bbm[0],axis=1)) ####################PriorityQueuingMethod()
        #     sslstm_mak_bbm = new_pro_ga.GetMakespan()
        #     print(sslstm_mak_bbm)
        new_pro_ga.PlotResult()

    return R, P

    def main(R):
        m = 5
        n = 5
        #     training_model(features,labels)

        # defined the new JSSP instance
        new_m = int(R.shape[1])
        new_n = int(R.shape[0])
        solving_new_jssp(new_m, new_n)

    if __name__ == "__main__":
        R, P = ss_lstm(request)
        main(R)
