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

local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
    -- a weight with points extracted from a normal distribution of mean 0.0 and co-variance .02 irrespective of dimensions
      m.weight:normal(0.0, 0.02)
    -- a bias with only zero as a number irrespective of dimensions 
      m.bias:fill(0)
-- 
-- SOMETHING CALLED BATCHNORMALIZATION 
-- 
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end


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
    which_epoch = 'latest',            -- which epoch to test? set to 'latest' to use latest cached model
    lr = 0.0002,            -- initial learning rate for adam
    beta1 = 0.5,            -- momentum term of adam
    condition_GAN = 1,
    lambda = 1000,
    use_GAN = 1
}

local ngf =3 
local ndf =3 
-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
opt.nThreads = 1 -- test only works with 1 thread...
print(opt)
if opt.display == 0 then opt.display = false end

local real_label = 1
local fake_label = 0

-- Criterias for the networks
local criterion = nn.BCECriterion()
local criterionAE = nn.AbsCriterion()

optimStateG = {
   -- learning rate of the optimizer
   learningRate = opt.lr,
   -- momentum of the adam optimizer
   beta1 = opt.beta1,
}

opt.manualSeed = torch.random(1, 10000) -- set seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setdefaulttensortype('torch.FloatTensor')

opt.netG_name = opt.name .. '/' .. opt.which_epoch .. '_net_G'
opt.netD_name = opt.name .. '/' .. opt.which_epoch .. '_net_D'


local data_loader = paths.dofile('data/data.lua')
print('#threads...' .. opt.nThreads)
local data = data_loader.new(opt.nThreads, opt)
print("Dataset Size: ", data:size())

-- translation direction
local idx_A = nil
local idx_B = nil
local errG = 0
local input_nc = opt.input_nc
local output_nc = opt.output_nc
if opt.which_direction=='AtoB' then
  idx_A = {1, input_nc}
  idx_B = {input_nc+1, input_nc+output_nc}
elseif opt.which_direction=='BtoA' then
  idx_A = {input_nc+1, input_nc+output_nc}
  idx_B = {1, input_nc}
else
  error(string.format('bad direction %s',opt.which_direction))
end
----------------------------------------------------------------------------

local input = torch.FloatTensor(opt.batchSize,3,opt.fineSize,opt.fineSize)
local target = torch.FloatTensor(opt.batchSize,3,opt.fineSize,opt.fineSize)
local Generator_out = torch.FloatTensor(opt.batchSize,3,opt.fineSize,opt.fineSize)
local fake_AB = torch.Tensor(opt.batchSize, output_nc + input_nc*opt.condition_GAN, opt.fineSize, opt.fineSize)

print('checkpoints_dir', opt.checkpoints_dir)
-- print('checkpoints_dir', opt.checkpoints_dir)
local netG = util.load(paths.concat(opt.checkpoints_dir, opt.netG_name .. '.t7'), opt)
local netD = util.load(paths.concat(opt.checkpoints_dir, opt.netD_name .. '.t7'), opt)
-- local netG = defineG_unet(input_nc, output_nc, ngf)
-- local netD = defineD_basic(input_nc, output_nc, ndf)
-- -- netG:apply(weights_init)
-- -- netD:apply(weights_init)
local parametersG, gradParametersG = netG:getParameters()
print(netG)
-- takes in number of input/output channels  and number of filters in first convo layer


function TableConcat(t1,t2)
    for i=1,#t2 do
        t1[#t1+1] = t2[i]
    end
    return t1
end


if opt.gpu > 0 then
   print('transferring to gpu...')
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   input = input:cuda();
   target = target:cuda(); Generator_out = Generator_out:cuda(); 
   fake_AB = fake_AB:cuda();
   if opt.cudnn==1 then
      netG = util.cudnn(netG); netD = util.cudnn(netD);
   end
   netD:cuda(); netG:cuda(); criterion:cuda(); criterionAE:cuda();
   print('done')
else
    print('running model on CPU')
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
    netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    
    gradParametersG:zero()
    
    -- GAN loss
    local df_dg = torch.zeros(Generator_out:size())
    if opt.gpu>0 then 
        df_dg = df_dg:cuda();
    end
    
    if opt.use_GAN==1 then
       local output = netD:forward(fake_AB)
       local label = torch.FloatTensor(output:size()):fill(real_label) -- fake labels are real for generator cost as we need to minimize the cost  when output-label is done
       if opt.gpu>0 then 
        label = label:cuda();
       end
       errG = criterion:forward(output, label)
       
       local df_do = criterion:backward(output, label) -- grad wrt the output of the networks
       df_dg = netD:updateGradInput(fake_AB, df_do):narrow(2,fake_AB:size(2)-output_nc+1, output_nc) -- update the gradients wrt the input
    else
        errG = 0
    end
    
    -- unary loss
    local df_do_AE = torch.zeros(Generator_out:size())
    if opt.gpu>0 then 
        df_do_AE = df_do_AE:cuda();
    end
    if opt.use_L1==1 then
       errL1 = criterionAE:forward(Generator_out, target)
       df_do_AE = criterionAE:backward(Generator_out, target)
    else
        errL1 = 0
    end
    
    netG:backward(input, df_dg + df_do_AE:mul(opt.lambda))
    
    return errG, gradParametersG
end



if opt.how_many=='all' then
    opt.how_many=data:size()
end
opt.how_many=math.min(opt.how_many, data:size())

local filepaths = {} -- paths to images tested on
local errors = {}
for n=1,math.floor(opt.how_many/opt.batchSize) do
    print('processing batch ' .. n)
    
    local data_curr, filepaths_curr = data:getBatch()
    filepaths_curr = util.basename_batch(filepaths_curr)
    print('filepaths_curr: ', filepaths_curr)
    
    input = data_curr[{ {}, idx_A, {}, {} }]
    target = data_curr[{ {}, idx_B, {}, {} }]
    -- -- Generator_out = netG:forward(input)
    
    --       netOut = netG:get(63).output      

    --       disp.image(util.scaleBatch(noise_out,100,100),{win=opt.display_id, title='noise_out'})
    if opt.gpu > 0 then
        input = input:cuda()
    end
    noise = torch.zeros(1,1,16,16)
    local y = 1
    local z = 1

    while y<17 do
      z=1
      while z<17 do 
        noise[{1,1,z,y}] = torch.uniform()*256
        z = z+1
       end 
      y = y + 1
    end  
    if opt.gpu>0 then 
      noise = noise:cuda();
    end
    -- print(sample:size())
    -- created fake using real_A as input
    -- fake_B = netG:forward({n, input})
    
    -- input_noisy = torch.cat(input,sample,4)
    -- print(real_A:size())
    -- print(sample:size())
    -- created fake using real_A as input
    
    -- -- print(input:size())
    -- local netOut = torch.FloatTensor(opt.batchSize,3,opt.fineSize,opt.fineSize)
    -- netOut = Generator_out:cuda();

    -- local noise_out = torch.FloatTensor(opt.batchSize,3,opt.fineSize,opt.fineSize)
    -- noise_out = Generator_out:cuda();

    -- local noise_out_old = torch.FloatTensor(opt.batchSize,3,opt.fineSize,opt.fineSize)
    -- -- noise_out_old = Generator_out:cuda();

    -- local gen_out_old = torch.FloatTensor(opt.batchSize,3,opt.fineSize,opt.fineSize)
    -- -- gen_out_old = Generator_out:cuda();

    if opt.preprocess == 'colorization' then
       local output_AB = netG:forward({n, input}):float()
       local input_L = input:float() 
       output = util.deprocessLAB_batch(input_L, output_AB)
       local target_AB = target:float()
       target = util.deprocessLAB_batch(input_L, target_AB)
       input = util.deprocessL_batch(input_L)
    else 
        if (n==1)  then
          print("Hello from the inside")
          Generator_out = netG:forward({noise, input})
          Generator_out_f = Generator_out:float()
          netOut = netG:get(65).output
          gen_out_old = Generator_out_f
          noise_out = util.normalize(netOut:float())
          noise_out_old = noise_out
        -- optim.adam(fGx, parametersG, optimStateG)
        else
          Generator_out = netG:forward({noise, input})
          Generator_out_f = Generator_out:float()
          netOut = netG:get(65).output      
          noise_out = util.normalize(netOut:float())
          a = noise_out-noise_out_old
          b = Generator_out_f-gen_out_old
          disp.image(util.scaleBatch(a,100,100),{win=opt.display_id, title='noise difference'})
          disp.image(util.scaleBatch(b,100,100),{win=opt.display_id+1, title='Generator difference'})
          noise_out_old = noise_out
          gen_out_old = Generator_out_f
        end 

        -- netOut = netG:get(63).output
        -- noise_out = util.normalize(netOut:float())
        -- print(noise_out)
                 
        fake_AB = torch.cat(input,Generator_out,2)
        output = util.deprocess_batch(Generator_out)
        noise_out = util.deprocess_batch(noise_out)
        input = util.deprocess_batch(input):float()
        output = output:float()
        noise_out = noise_out:float()
        target = util.deprocess_batch(target):float()
    end
    paths.mkdir(paths.concat(opt.results_dir, opt.netG_name .. '_' .. opt.phase))
    local image_dir = paths.concat(opt.results_dir, opt.netG_name .. '_' .. opt.phase, 'images')
    paths.mkdir(image_dir)
    paths.mkdir(paths.concat(image_dir,'noise_out'))
    paths.mkdir(paths.concat(image_dir,'input'))
    paths.mkdir(paths.concat(image_dir,'output'))
    paths.mkdir(paths.concat(image_dir,'target'))
    -- print(input:size())
    -- print(output:size())
    -- print(target:size())
    for i=1, opt.batchSize do
        image.save(paths.concat(image_dir,'noise_out',filepaths_curr[i]), image.scale(noise_out[i],noise_out[i]:size(2),noise_out[i]:size(3)/opt.aspect_ratio))      
        image.save(paths.concat(image_dir,'input',filepaths_curr[i]), image.scale(input[i],input[i]:size(2),input[i]:size(3)/opt.aspect_ratio))
        image.save(paths.concat(image_dir,'output',filepaths_curr[i]), image.scale(output[i],output[i]:size(2),output[i]:size(3)/opt.aspect_ratio))
        image.save(paths.concat(image_dir,'target',filepaths_curr[i]), image.scale(target[i],target[i]:size(2),target[i]:size(3)/opt.aspect_ratio))
    end
    print('Saved images to: ', image_dir)
    
    if opt.display then
      if opt.preprocess == 'regular' then
        disp = require 'display'
        -- print(1)
        disp.image(util.scaleBatch(noise_out,100,100),{win=opt.display_id+2, title='noise_out'})
        -- print(2)
        -- disp.image(util.scaleBatch(noise_out:float(),100,100),{win=opt.display_id, title=opt.name .. 'noise_out'})

        disp.image(util.scaleBatch(input,100,100),{win=opt.display_id+3, title='input'})
        disp.image(util.scaleBatch(output,100,100),{win=opt.display_id+4  , title='output'})
        disp.image(util.scaleBatch(target,100,100),{win=opt.display_id+5, title='target'})
        
        print('Displayed images')
      end
    end
    
    filepaths = TableConcat(filepaths, filepaths_curr)
    errors[#errors+1] = errG

    print(('Err_G: %.4f' ):format(errG))
end

-- make webpage
io.output(paths.concat(opt.results_dir,opt.netG_name .. '_' .. opt.phase, 'index.html'))

io.write('<table style="text-align:center;">')

io.write('<tr><td>Image #</td><td>Input</td><td>Output</td><td>Ground Truth</td></tr>')
for i=1, #filepaths do
    io.write('<tr>')
    io.write('<td>' .. filepaths[i] .. '</td>')
    io.write('<td>' .. errors[i] .. '</td>')
    io.write('<td><img src="./images/noise_out/' .. filepaths[i] .. '"/></td>')
    io.write('<td><img src="./images/input/' .. filepaths[i] .. '"/></td>')
    io.write('<td><img src="./images/output/' .. filepaths[i] .. '"/></td>')
    io.write('<td><img src="./images/target/' .. filepaths[i] .. '"/></td>')
    io.write('</tr>')
end

io.write('</table>')