# EECS 442 Project
## Our code
### Files
1. unsup_alexnet.pth: the model weights we trained on COCO using alexnet
2. this github: pytorch version of faster RCNN with additional alexnet (lib/model/faster_rcnn/alexnet.py)

### Goal
1. Train Faster R-CNN with alexnet on Pascal VOC (with loading unsup_alexnet.pth)


## Their code
Files from unsupervised_video
### Files
#### Transfer_learning Model
1. .t7 file (used for lua torch): we can't use this directly. Use the corresonding .pth and .py (in folder ./442/)
- I convert this t7 to pth and an associated CNN structure (stored in the corresonding py file)
- This file should work on pytorch 0.4.0 (no higher)
2. Train faster r-cnn with this model on Pascal VOC with loading the corresonding .pth
3. unsup_video.pth/.py is the one you should first try.

## Procedures
1. Write code you think is necessary
2. Run our code
3. If this works (you can train it and the visualization (or loss) looks not bad), try their code
4. If both works, you run one of the above two codes and leave the other one to me
5. If one doesn't work and you can't figure it out, tell me and we can chat via zoom