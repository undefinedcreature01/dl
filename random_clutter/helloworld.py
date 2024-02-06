
import torch
import pandas


tensor32_A = torch.rand(2,3)
tensor32_B = torch.rand(2,3)

tensor32_C = torch.tensor(
    [
        [1,2,3],
        [4,5,6]
    ]
)

tensor32_D = torch.tensor(
    [
        [1,2,3],
        [4,5,6]
    ]
)

tensor32_E = torch.tensor(
    [
        [1,2],
        [3,4],
        [5,6]
    ]
)


#print(torch.matmul(tensor32_C, tensor32_D.T))
#print(torch.mean(tensor32_C))


some_tensor = torch.rand(10)

print(some_tensor.device)
