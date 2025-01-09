import time
from mcd_with_vi import torch_tutorial_MCD
from standard_fully_connected import torch_standard
from sample_parallel_mcd import sample_parallel_mcd
from mcd_regular import torch_MCD_regular
import torch
from mpi4py import MPI
from parallel_mcd import parallel_mcd
from parallel_mcd import mnist_parallel_mcd
from regular_mcd import regular_mcd
from regular_mcd import mnist_regular_mcd
from regular_no_mcd import regular_linear
from regular_no_mcd import mnist_regular_linear
from vi.linear import sampling
import pickle
import csv


if __name__ == "__main__":
    #comm = MPI.COMM_WORLD
    #rank = comm.Get_rank()
    file = open("no_mcd_42.pkl", "rb")
    model = pickle.load(file)

    global sampling
    last_len = 0
    time_stamps = []
    input_length = 50
    output_length = 10
    hidden1 = 40
    hidden2 = 20
    batch_sizes = [32]
    epochs = 15
    general_seeds = [42]
    sample_nums = [200]
    lrs = [1e-3, 1e-4, 1e-5]
    init_stds = [2]
    id = 0
    torch.manual_seed(61)
    #58, 59
    #regular_linear(input_length, hidden1, hidden2, output_length, 128, epochs, 0, model)
    regular_mcd(input_length, hidden1, hidden2, output_length, 32, epochs, 200, 2)
    '''
    regular_mcd(input_length, hidden1, hidden2, output_length, 32, epochs, 200, 2, model)
    data = [
        ["id", "random_seed", "batch_size", "sample_num", "init_std", "time", "loss", "epoch_num"]
    ]
    #print(rank)
    for general_seed in general_seeds:
        torch.manual_seed(general_seed)
        for batch_size in batch_sizes:
            for sample_num in sample_nums:
                for init_std in init_stds:
                    filename = "reg_mcd_" + str(id) + ".pkl"
                    start_time = time.time()
                    model, loss, epoch_final = regular_mcd(input_length, hidden1, hidden2, output_length, batch_size, epochs, sample_num, init_std)
                    script_time = start_time - time.time()
                    with open(filename, 'wb') as file:
                        pickle.dump(model, file)
                    data.append([id, general_seed, batch_size, sample_num, init_std, script_time, loss, epoch_final])
                    id += 1

    with open("reg_mcd_different.csv", "w", newline="") as file:
        writer = csv.writer(file)

        # Write each row into the file
        writer.writerows(data)

    '''
    '''
    for i in range(5):
        regular_mcd(input_length, hidden1, hidden2, output_length, batch_size, epochs, sample_num)
        time_lapsed = sum(sampling[last_len:])
        time_stamps.append(time_lapsed)
        last_len = len(sampling)
        sample_num *= 10

    print(time_stamps)

    '''
    #mnist_regular_linear()
    # mnist_parallel_mcd()
    #parallel_mcd(input_length, hidden1, hidden2, output_length, batch_size, epochs, sample_num)
    #torch_MCD_regular(input_length, hidden1, hidden2, output_length, batch_size, epochs)
    #start_time = time.time()

    #torch_tutorial_MCD(input_length, hidden1, hidden2, output_length, batch_size, epochs)

    #print("--- %s seconds ---" % (time.time() - start_time))
    #start_time = time.time()
    #torch_standard(input_length, hidden1, hidden2, output_length, batch_size, epochs)
    #print("--- %s seconds ---" % (time.time() - start_time))


    #[8.241406917572021, 13.140380144119263, 56.135470151901245, 504.34021067619324, 4446.123069286346]
