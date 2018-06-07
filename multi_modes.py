import subprocess
import os
import sys
import cv2
import numpy as np

my_env = os.environ
def bash(command):
    p = subprocess.Popen(command, env=my_env, shell=True)
    pd = p.pid
    p.wait()
    # process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    # output, error = process.communicate()
    # return output,error
opt = {
    'Model' : None,
    'images' : [],
}



transform = 'BtoA'
alpha = 10 # High means low impact of adverserial
lbda = 1 # High means high impact of adverserial
niter = 1000

ntrain = 2

# if(sys.argv[1].split('.') != 'png'):
opt['Model'] = sys.argv[1]
opt['seq_len'] = int(sys.argv[2])
opt['Dataset'] = sys.argv[3] + '/latest_net_G_val/images/output'
start_img = int(sys.argv[4])
gap = int(sys.argv[5])

# else:
#     opt['images'].append(sys.argv[1])

for i in range(ntrain):
    # 'data_mode/' + 
    name  = opt['Dataset'] + '/img_' + str(start_img + i*gap) + '.png'
    print(name)
    opt['images'].append(name)


name = opt['Model']
continue_tr = 0
if opt['Model'] is not None:
    continue_tr = 1
    if(name.split('_')[-1] != 'normal'):
        bash('cp -r checkpoints/' + name + ' checkpoints/' + name + '_normal' )
        bash('mv checkpoints/' + name + ' checkpoints/' + name + '_mmodal' )
        opt['Model'] = name + '_mmodal' #something
    else:
        nm2 = name.split('_')
        name = '_'.join(nm2[:-1])
        bash('cp -r checkpoints/' + name + '_normal' + ' checkpoints/' + name + '_mmodal' )
        opt['Model'] = name + '_mmodal'
        

# for i in range(2,len(sys.argv)-1):
#     opt['images'].append(sys.argv[i])


bash("rm train_chain/train/*")
bash("rm train_chain/val/*")

# for img in opt['images']:
#     name = img.split('/')[-1]
#     bash('cp ' + img + ' train_chain/train/' + name)

name_1 = opt['images'][0].split('/')[-1]
for i in range(1, len(opt['images'])):
    name_2 = opt['images'][i].split('/')[-1]
    
    # Move ahead.
    im_A = cv2.imread(opt['images'][i-1])#, cv2.CV_LOAD_IMAGE_COLOR)
    im_B = cv2.imread(opt['images'][i])#, cv2.CV_LOAD_IMAGE_COLOR)
    im_AB = np.concatenate([im_A, im_B], 1)
    path_AB = 'train_chain/train/' + name_1.split('.')[0] + '_' + name_2
    cv2.imwrite(path_AB, im_AB)
    name_1 = name_2

img = opt['images'][0]
name = img.split('/')[-1]

# bash('cp ' + img + ' train_chain/val/' + name)

im = cv2.imread(img)#, cv2.CV_LOAD_IMAGE_COLOR)
im_B = np.zeros(np.shape(im))
im_AB = np.concatenate([im, im_B], 1)
path_AB = 'train_chain/val/' + name.split('.')[0] + '_' + 'black.png'
cv2.imwrite(path_AB, im_AB)
name_1 = name_2



bash('mkdir results_chain/' + opt['Model'])
bash('cp ' + img + ' results_chain/' + opt['Model'] + '/' + name.split('.')[0] + '_' + 'black_' + str(start_img) + '.png')


cmd = 'DATA_ROOT=./train_chain  name={0} lambda={1} alpha={2} which_direction={3} continue_train={4} niter={5} th train.lua'.format(\
    opt['Model'],lbda,alpha,transform, continue_tr, niter)

print(cmd)
print('# -------- Training ---------------')

bash(cmd)

# CUDA_VISIBLE_DEVICES=3
cmd = ' DATA_ROOT=./train_chain  name={0} lambda={1} alpha={2} which_direction=AtoB th test.lua'.format(opt['Model'],lbda,alpha)

ctc_img = cv2.imread('results_chain/' + opt['Model'] + '/' + name.split('.')[0] + '_' + 'black_' +str(start_img) + '.png')
prefix = '/'.join(opt['images'][0].split('/')[:-2])
ctc_gt = cv2.imread(prefix + '/target/' + 'img_' + str(gap) + '.png')
print(prefix + '/target/' + 'img_' + str(gap) + '.png')
ctc_cg = cv2.imread(prefix + '/output/' + 'img_' + str(gap) + '.png')
ctc_rgb = cv2.imread(prefix + '/input/' + 'img_' + str(gap) + '.png')

for i in range(opt['seq_len']):
    bash(cmd)
    bash('cp results/' + opt['Model'] + '/latest_net_G_val/images/output/'  + name.split('.')[0] + '_' + 'black.png' + ' ' + 'results_chain/' + opt['Model'] + '/' + name.split('.')[0] + '_' + 'black_' + str(i+1) + '.png')
    img = 'results/' + opt['Model'] + '/latest_net_G_val/images/output/' + \
        name.split('.')[0] + '_' + 'black.png'
    
    print(img)
    im = cv2.imread(img)#, cv2.CV_LOAD_IMAGE_COLOR)
    im_B = np.zeros(np.shape(im))
    im_AB = np.concatenate([im, im_B], 1)
    ctc_img = np.concatenate([ctc_img, im], 1)
    ig = cv2.imread(prefix + '/target/' + 'img_' + str(start_img + i*gap+gap) + '.png')
    ctc_gt = np.concatenate([ctc_gt, ig], 1)
    ig = cv2.imread(prefix + '/output/' + 'img_' + str(start_img + i*gap+gap) + '.png')
    ctc_cg = np.concatenate([ctc_cg, ig], 1)
    ig = cv2.imread(prefix + '/input/' + 'img_' + str(start_img + i*gap+gap) + '.png')
    ctc_rgb = np.concatenate([ctc_rgb, ig], 1)

    path_AB = 'train_chain/val/' + name.split('.')[0] + '_' + 'black.png'
    cv2.imwrite(path_AB, im_AB)


ctc_img = np.concatenate([ctc_img, ctc_rgb], 0)
ctc_img = np.concatenate([ctc_img, ctc_gt], 0)
ctc_img = np.concatenate([ctc_img, ctc_cg], 0)

cv2.imwrite('results_chain/' + opt['Model'] + '/' + name.split('.')[0] + '_strip.png', ctc_img)




print('# -------- Testing ---------------')


# Ending
