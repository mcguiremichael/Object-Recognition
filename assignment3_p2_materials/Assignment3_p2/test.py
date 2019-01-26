

import torch
from yolo_loss import YoloLoss
torch.manual_seed(2)
torch.cuda.manual_seed(2)
torch.backends.cudnn.deterministic = True

criterion = YoloLoss(14, 2, 5, 0.5)


for i in range(100):
    pred = torch.rand((10, 14, 14, 30)).cuda()
    target = torch.rand((10, 14, 14, 30)).cuda()
    loss = criterion(pred, target)
    print(loss)
    
    






