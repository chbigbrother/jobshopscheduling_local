import pandas as pd
import numpy as np

def generate_jssp_instance_from_prediction(pre):
    f = open('./schedule/myfile.txt', 'w')
    a = pre.sort_values(by="machine ID")
    job_numbers = len(a["name of product"].unique())
    machines_numbers = a["machine ID"].max()
    avg = 1

    job_names = a["name of product"].unique()
    each_job_tasks = a.groupby(["name of product", "machine ID"]).size().unstack().count(axis=1)

    b = a.groupby(["name of product", "machine ID"]).count()
    array_length = b.unstack()["machine numbers"].sum(axis=1) * 2 + each_job_tasks + 1

    f.write('%d %d %d\n' % (job_numbers, machines_numbers, avg))

    start_sublength = 0
    sub_length = each_job_tasks[0]

    for j in range(job_numbers):
        array = np.zeros(array_length[j].astype(int))
        sub_operation_numbers = each_job_tasks[j]
        array[0] = sub_operation_numbers

        # extract the operation numbers and corresponding machines
        c = b.unstack()["machine numbers"].iloc[j,].dropna()
        sub_operation_numbers_each_job = c.values

        required_machine = c.index
        #     print(sub_operation_numbers_each_job)

        # processing time and corresponding machine
        d = a[b.index.levels[0][j] == a["name of product"]].loc[:, ["machine ID", "total processing time"]].values

        # fill the array
        i = 0  # the suboperation index
        k = 0  # the processing time index
        z = 1  # the array index
        for i in range(len(sub_operation_numbers_each_job)):
            array[z] = sub_operation_numbers_each_job[i]
            array[(z + 1):(z + 1 + 2 * sub_operation_numbers_each_job[i].astype(int))] = d[k:(
                        k + sub_operation_numbers_each_job[i].astype(int))].reshape(1, (
                        2 * sub_operation_numbers_each_job[i]).astype(int))

            k = k + sub_operation_numbers_each_job[i].astype(int)
            # print(k)
            #         k=sub_operation_numbers_each_job[i].astype(int)
            z = z + 2 * sub_operation_numbers_each_job[i].astype(int) + 1
            #         print(z,k,d[k:(k+sub_operation_numbers_each_job[i].astype(int))].reshape(1,(2*sub_operation_numbers_each_job[i]).astype(int)))
            i = i + 1
        np.savetxt(f, array.reshape(1, array.shape[0]), fmt="%d")
    f.close()

# if __name__ == '__main__':
#     pre=pd.read_csv("./prediction.csv")
#     generate_jssp_instance_from_prediction(pre)

