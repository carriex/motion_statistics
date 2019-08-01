import torch.optim as optim 
import torch.nn as nn 
import torch
import model
import os
import numpy as np
from dataset import UCF101

test_list = 'list/test_ucf101.list'
batch_size = 12
num_classes = 101 
model_dir = 'models'
model_name = 'c3d-finetune.pth-60000'

def eval():
	
	model_path = os.path.join(model_dir,model_name)
	device = get_default_device()
	c3d = model.C3D(num_classes)
	c3d.load_state_dict(torch.load(model_path))
	c3d.to(device, non_blocking=True,dtype=torch.float)
	c3d.eval()

	testset = UCF101(datalist_file=test_list, clip_len=16, crop_size=112,split="testing")
	testloader = torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=True,num_workers=4) 

	total_predict_label = []
	total_accuracy = [] 

	for (i, data) in enumerate(testloader, 0):
		inputs, labels = data['clip'].to(device,dtype=torch.float), data['label'].to(device)
		_, outputs = c3d(inputs).max(1)
		total = labels.size(0)
		correct = (outputs == labels).sum().item()
		accuracy = float(correct) / float(total)
		print("iteration %d, accuracy = %g" % (i, accuracy))
		total_predict_label.append(outputs)
		total_accuracy.append(accuracy)

	total_accuracy = np.array(total_accuracy)
	total_predict_label = np.array(total_predict_label)


def get_default_device():
	if torch.cuda.is_available():
		return torch.device('cuda:0')
	else:
		return torch.device('cpu')



def main():
	eval()

if __name__ == "__main__":
	main()



