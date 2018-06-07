require 'distributions'
require 'torch'
require 'image'
require 'optim'
require 'nn'
require 'nngraph'
util = paths.dofile('util/util.lua')
torch.setdefaulttensortype('torch.FloatTensor')
require 'models'
mu = torch.zeros(100)
sigma = 100*torch.eye(100)
n = torch.zeros(1,3,100,100)
real_A = torch.zeros(1,3,256,256)

-- local i = 1
-- while i<101 do
-- 	sample_1 = distributions.mvn.rnd(mu, sigma) -- a 
-- 	sample_2 = distributions.mvn.rnd(mu, sigma) -- a 
-- 	sample_3 = distributions.mvn.rnd(mu, sigma) -- a 
-- 	n[{1,1,{1,100},i}] = sample_1
-- 	n[{1,2,{1,100},i}] = sample_2
-- 	n[{1,3,{1,100},i}] = sample_3
--   i = i + 1
-- end
ngf=64
output_nc=3
local n1 = - nn.SpatialFullConvolution(1, ngf, 4, 4, 2, 2, 1, 1)
-- input is (ngf) x 2 x 2
local n2 = n1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf , ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
-- input is (ngf * 2) x 4 x 4
local n3 = n2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)  --62
-- input is (ngf * 4) x 8 x 8
local n4 = n3 - nn.LeakyReLU(0.2, true) - nn.SpatialFullConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
-- input is (ngf * 8) x 16 x 16
local n5 = n4 - nn.LeakyReLU(0.2, true) - nn.SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
-- input is (ngf * 8) x 32 x 32
local n6 = n5 - nn.LeakyReLU(0.2, true) - nn.SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
-- input is (ngf * 4) x 64 x 64
local n7 = n6 - nn.LeakyReLU(0.2, true) - nn.SpatialFullConvolution(ngf * 2, ngf * 1, 4, 4, 2, 2, 1, 1)  - nn.SpatialBatchNormalization(ngf * 1)
-- input is (ngf * 2) x 128 x 128
local n8 = n7 - nn.LeakyReLU(0.2, true) - nn.SpatialFullConvolution(ngf * 1, output_nc, 4, 4, 2, 2, 1, 1) --76
-- input is (output_nc ) x 256 x 256

netG = nn.gModule({n1},{n8})

noise = torch.zeros(1,1,1,1)

noise[{1,1,1,1}] = torch.uniform()*255
-- print(real_A:size())

netOut = netG:forward(noise)

noise_out = util.normalize(netOut:float())
noise_out = util.deprocess_batch(noise_out)
noise_out = noise_out:float()
-- paths.mkdir(paths.concat(image_dir,'noise_out'))
disp = require 'display'
disp.image(util.scaleBatch(noise_out,100,100),{win=200, title='noise_out'})
