-- usage: DATA_ROOT=/path/to/data/ name=expt1 which_direction=BtoA th test.lua
--
-- code derived from https://github.com/soumith/dcgan.torch
--
require 'torch'
require 'image'
require 'optim'
require 'nn'
require 'nngraph'
util = paths.dofile('util/util.lua')
torch.setdefaulttensortype('torch.FloatTensor')
require 'models'
-- require 'distributions'

opt = {
    DATA_ROOT = '',           -- path to images (should have subfolders 'train', 'val', etc)
    batchSize = 1,            -- # images in batch
    loadSize = 256,           -- scale images to this size
    fineSize = 256,           --  then crop to this size
    flip=0,                   -- horizontal mirroring data augmentation
    display = 1,              -- display samples while training. 0 = false
    display_id = 200,         -- display window id.
    gpu = 1,                  -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
    how_many = 'all',         -- how many test images to run (set to all to run on every image found in the data/phase folder)
    which_direction = 'BtoA', -- AtoB or BtoA
    phase = 'val',            -- train, val, test ,etc
    preprocess = 'regular',   -- for special purpose preprocessing, e.g., for colorization, change this (selects preprocessing functions in util.lua)
    aspect_ratio = 1.0,       -- aspect ratio of result images
    name = '',                -- name of experiment, selects which model to run, should generally should be passed on command line
    input_nc = 3,             -- #  of input image channels
    output_nc = 3,            -- #  of output image channels
    serial_batches = 1,       -- if 1, takes images in order to make batches, otherwise takes them randomly
    serial_batch_iter = 1,    -- iter into serial image list
    cudnn = 1,                -- set to 0 to not use cudnn (untested)
    checkpoints_dir = './checkpoints', -- loads models from here
    results_dir='./results/',          -- saves results here
    which_epoch = '500',            -- which epoch to test? set to 'latest' to use latest cached model
    lr = 0.0002,            -- initial learning rate for adam
    beta1 = 0.5,            -- momentum term of adam
    condition_GAN = 1,
    lambda = 1000,
    use_GAN = 1,
    which_model_netD = 'basic', -- selects model to use for netD
    which_model_netG = 'unet',  -- selects model to use for netG
    save = './weights/',
    layer=1,
    ngf = 64,               -- #  of gen filters in first conv layer
    ndf = 64,		
}


-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
opt.nThreads = 1 -- test only works with 1 thread...
print(opt)
if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- set seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setdefaulttensortype('torch.FloatTensor')

opt.netG_name = opt.name .. '/' .. opt.which_epoch .. '_net_G'
opt.netD_name = opt.name .. '/' .. opt.which_epoch .. '_net_D'

local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

function defineG(input_nc, output_nc, ngf)
    local netG = nil
    if     opt.which_model_netG == "encoder_decoder" then netG = defineG_encoder_decoder(input_nc, output_nc, ngf)
    elseif opt.which_model_netG == "unet" then netG = defineG_unet(input_nc, output_nc, ngf)
    elseif opt.which_model_netG == "unet_128" then netG = defineG_unet_128(input_nc, output_nc, ngf)
    else error("unsupported netG model")
    end
   
    netG:apply(weights_init)
  
    return netG
end

function defineD(input_nc, output_nc, ndf)
    local netD = nil
    if opt.condition_GAN==1 then
        input_nc_tmp = input_nc
    else
        input_nc_tmp = 0 -- only penalizes structure in output channels
    end
    
    if     opt.which_model_netD == "basic" then netD = defineD_basic(input_nc_tmp, output_nc, ndf)
    elseif opt.which_model_netD == "n_layers" then netD = defineD_n_layers(input_nc_tmp, output_nc, ndf, opt.n_layers_D)
    else error("unsupported netD model")
    end
    
    netD:apply(weights_init)
    
    return netD
end



print('checkpoints_dir', opt.checkpoints_dir)
local netG = defineG(opt.input_nc, opt.output_nc, opt.ngf)
--util.load(paths.concat(opt.checkpoints_dir, opt.netG_name .. '.t7'), opt)
local netD = defineD(opt.input_nc, opt.output_nc, opt.ndf)
--util.load(paths.concat(opt.checkpoints_dir, opt.netD_name .. '.t7'), opt)
local parametersG, gradParametersG = netG:getParameters()
print(netG)

local layer=1
for layer =1,100 do 
	print(layer) 
	--print(layer)
	--print(netG:get(layer).weight)
	if (netG:get(layer).weight ~=nil) then 
		local prod = 1
		local weights = netG:get(layer).weight:clone()
		for i=1, #weights:size() do
			prod = prod*weights:size()[i]
		end
		--print(prod)
		--print(weights:size()[1])
		--print(weights:size()[2])
		--print(weights:size()[3])
		--print(weights:size()[4])
		--if(weights:size()[2]*weights:size()[1]/3 == 0) then
		if(prod>65500) then y = torch.reshape(weights, torch.LongStorage{512,prod/512})
		else y = torch.reshape(weights, torch.LongStorage{64,prod/64}) end
		--else
		--	y =  torch.reshape(weights, torch.LongStorage{weights:size()[1]*weights:size()[2]/3,3, weights:size()[2], weights:size()[1]}) 
		--end
		--print(torch.median(weights):float())
		--print(torch.mode(weights):float())
		--print(torch.mean(weights):float())
		image.saveJPG(paths.concat(opt.save,'Initialized_Layer'..tostring(layer)..'Filters_'.. opt.name .. '.jpg'), image.toDisplayTensor(y))
		
	end
end




