import torch.optim as optim 
import torch.nn as nn 
import torch, torchvision
import model
import os
import time
from dataset import UCF101DataSet
from tensorboardX import SummaryWriter


base_lr = 0.001
momentum = 0.9 
batch_size = 30
num_classes = 14
num_epoches = 18
weight_decay = 0.005
framelist_file = 'list/rgb_train_linux.list'
v_flow_list_file='list/v_flow_train_linux.list'
u_flow_list_file='list/u_flow_train_linux.list'
clip_len = 16
model_dir = 'models'
model_name = 'c3d-motion-new.pth'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def train():

	#initialize the model
	c3d = model.C3D(num_classes,pretrain=True)

	train_param = [
					{'params':c3d.get_1x_lr_param()},
					{'params':c3d.get_2x_lr_param(), 'lr': base_lr*2}]

	device = get_default_device()
	print(device)

	print(c3d)

	#import input data
	trainset = UCF101DataSet(framelist_file=framelist_file, v_flow_list_file=v_flow_list_file, u_flow_list_file=u_flow_list_file,clip_len=clip_len, crop_size=112,split="training")
	trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=10)
	

	c3d.to(device, non_blocking=True,dtype=torch.float)
	c3d.train()

	#define loss function (MSE loss)
	criterion = nn.MSELoss()
	criterion.to(device)


	#define optimizer 
	optimizer = optim.SGD(train_param, lr=base_lr, momentum=momentum, weight_decay=weight_decay)

	#lr is divided by 10 after every 4 epoches 
	
	scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=6, gamma=0.1)

	writer = SummaryWriter()

	for epoch in range(num_epoches):
		
		running_loss = 0.0 
		scheduler.step() #last_epoch default is -1

		for i, data in enumerate(trainloader, 0):

			step = epoch * len(trainloader) + i
			inputs, labels = data['clip'].to(device,dtype=torch.float), data['label'].to(device)
			optimizer.zero_grad()

			
			outputs = c3d(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			
			optimizer.step()

			running_loss +=loss.item()
			print('Step %d, loss: %.3f' %(i, loss.item()))
			writer.add_scalar('Train/Loss', loss.item(),step)


			if i % 100 == 99:
				print('[%d, %5d] loss: %.3f' %
					(epoch + 1, i + 1, running_loss / 100))
				running_loss = 0.0 
			if step % 10000 == 9999:
				torch.save(c3d.state_dict(),os.path.join(model_dir,'%s-%d'%(model_name, step+1)))

	print('Finished Training')
	writer.close()

def get_default_device():
	if torch.cuda.is_available():
		return torch.device('cuda:0')
	else:
		return torch.device('cpu')



def main():
	train()

if __name__ == "__main__":
	main()





