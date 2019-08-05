import torch.optim as optim
import torch.nn as nn
import torch
import model
import os
import numpy as np
from dataset import UCF101DataSet
from utils import get_default_device
import constant

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

def eval():

    model_path = os.path.join(constant.MODEL_DIR, constant.TRAINED_MODEL)
    device = get_default_device()
    c3d = model.C3D(constant.NUM_CLASSES)
    c3d.load_state_dict(torch.load(model_path))
    c3d.to(device, non_blocking=True, dtype=torch.float)
    c3d.eval()
    print(model_path)


    testset = UCF101DataSet(framelist_file=constant.TEST_LIST,
                            clip_len=constant.CLIP_LENGTH,
                            crop_size=constant.CROP_SIZE, split="testing")
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=constant.TEST_BATCH_SIZE, shuffle=False, num_workers=8)

    total_predict_label = []
    total_accuracy = []

    for (i, data) in enumerate(testloader, 0):
        inputs, labels = data['clip'].to(
            device, dtype=torch.float), data['label'].to(device)
        _, outputs = c3d(inputs).max(1)

        total = labels.size(0)
        correct = (outputs == labels).sum().item()
        accuracy = float(correct) / float(total)
        print("iteration %d, accuracy = %g" % (i, accuracy))

        total_predict_label.append(outputs)
        total_accuracy.append(accuracy)

    total_accuracy = np.array(total_accuracy)
    total_predict_label = np.array(total_predict_label)
    
    print(model_path)
    print("Final accuracy", np.mean(total_accuracy))


def main():
    eval()


if __name__ == "__main__":
    main()
