import time
from mcd_with_vi import torch_tutorial_MCD
from standard_fully_connected import torch_standard
from mcd_regular import torch_MCD_regular
import torch


if __name__ == "__main__":
    input_length = 50
    output_length = 10
    hidden1 = 40
    hidden2 = 20
    batch_size = 32
    epochs = 10
    torch.manual_seed(42)

    #torch_MCD_regular(input_length, hidden1, hidden2, output_length, batch_size, epochs)
    #start_time = time.time()
    torch_tutorial_MCD(input_length, hidden1, hidden2, output_length, batch_size, epochs)
    #print("--- %s seconds ---" % (time.time() - start_time))
    #start_time = time.time()
    #torch_standard(input_length, hidden1, hidden2, output_length, batch_size, epochs)
    #print("--- %s seconds ---" % (time.time() - start_time))
