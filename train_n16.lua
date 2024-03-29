-- usage example: DATA_ROOT=/path/to/data/ which_direction=BtoA name=expt1 th train.lua 
--
-- code derived from https://github.com/soumith/dcgan.torch
--

require 'torch'
require 'nn'
require 'optim'
util = paths.dofile('util/util.lua')
require 'image'
require 'models_n16'
-- require 'distributions'
-- nngraph.setDebug(true)


opt = {
   DATA_ROOT = '',         -- path to images (should have subfolders 'train', 'val', etc)
   batchSize = 1,          -- # images in batch
   loadSize = 286,         -- scale images to this size
   fineSize = 256,         --  then crop to this size
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   input_nc = 3,           -- #  of input image channels
   output_nc = 3,          -- #  of output image channels
   niter = 100,            -- #  of iter at starting learning rate
   lr = 0.0002,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   flip = 1,               -- if flip the images for data argumentation
   display = 1,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = '',              -- name of the experiment, should generally be passed on the command line
   which_direction = 'AtoB',    -- AtoB or BtoA
   phase = 'train',             -- train, val, test, etc
   preprocess = 'regular',      -- for special purpose preprocessing, e.g., for colorization, change this (selects preprocessing functions in util.lua)
   nThreads = 2,                -- # threads for loading data
   save_epoch_freq = 100,        -- save a model every save_epoch_freq epochs (does not overwrite previously saved models)
   save_latest_freq = 500,     -- save the latest model every latest_freq sgd iterations (overwrites the previous latest model)
   print_freq = 50,             -- print the debug information every print_freq iterations
   display_freq = 2,          -- display the current results every display_freq iterations
   save_display_freq = 452,    -- save the current display of results every save_display_freq_iterations
   continue_train		=0,            -- if continue training, load the latest model: 1: true, 0: false
   serial_batches = 0,          -- if 1, takes images in order to make batches, otherwise takes them randomly
   serial_batch_iter = 1,       -- iter into serial image list
   checkpoints_dir = './checkpoints', -- models are saved here
   cudnn = 1,                         -- set to 0 to not use cudnn
   condition_GAN = 1,                 -- set to 0 to use unconditional discriminator
   use_GAN = 1,                       -- set to 0 to turn off GAN term
   use_L1 = 1,                        -- set to 0 to turn off L1 term
   which_model_netD = 'basic', -- selects model to use for netD
   which_model_netG = 'unet',  -- selects model to use for netG
   n_layers_D = 0,             -- only used if which_model_netD=='n_layers'
   lambda = 100,               -- weight on L1 term in objective INITIAL = 100
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

--  Saving the number of input channels and output channels
local input_nc = opt.input_nc
local output_nc = opt.output_nc
-- translation direction
local idx_A = nil
local idx_B = nil

if opt.which_direction=='AtoB' then
    idx_A = {1, input_nc}
    idx_B = {input_nc+1, input_nc+output_nc}
elseif opt.which_direction=='BtoA' then
    idx_A = {input_nc+1, input_nc+output_nc}
    idx_B = {1, input_nc}
else
    error(string.format('bad direction %s',opt.which_direction))
end

if opt.display == 0 then opt.display = false end


-- seedig to deetermine order of inputs
opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local data_loader = paths.dofile('data/data.lua')
print('#threads...' .. opt.nThreads)
local data = data_loader.new(opt.nThreads, opt)
print("Dataset Size: ", data:size())
tmp_d, tmp_paths = data:getBatch()

----------------------------------------------------------------------------
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

-- number of filters in first convo layer (64 by default)
local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 1
local fake_label = 0

-- takes in number of input/output channels  and number of filters in first convo layer
function defineG(input_nc, output_nc, ngf)
    local netG = nil
    if     opt.which_model_netG == "encoder_decoder" then netG = defineG_encoder_decoder(input_nc, output_nc, ngf)
-- generator uses the "unet" model by default      
    elseif opt.which_model_netG == "unet" then netG = defineG_unet(input_nc, output_nc, ngf)
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
-- discriminator uses the "basic" model by default          
    if     opt.which_model_netD == "basic" then netD = defineD_basic(input_nc_tmp, output_nc, ndf)
    elseif opt.which_model_netD == "n_layers" then netD = defineD_n_layers(input_nc_tmp, output_nc, ndf, opt.n_layers_D)
    else error("unsupported netD model")
    end
    
    netD:apply(weights_init)
    
    return netD
end


-- load saved models and finetune
if opt.continue_train == 1 then
   print('loading previously trained netG...')
   netG = util.load(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_G.t7'), opt)
   print('loading previously trained netD...')
   netD = util.load(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_D.t7'), opt)
else
  print('define model netG...')
  netG = defineG(input_nc, output_nc, ngf)
  print('define model netD...')
  netD = defineD(input_nc, output_nc, ndf)
end

print(netG)
print(netD)

-- Criterias for the networks
local criterion = nn.BCECriterion()
local criterionAE = nn.AbsCriterion()
---------------------------------------------------------------------------
optimStateG = {
   -- learning rate of the optimizer
   learningRate = opt.lr,
   -- momentum of the adam optimizer
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
----------------------------------------------------------------------------
--   a tensor for 'batchSize' number of images with input_nc channels which is used to condition the gans (fineSize)X(fineSize)
local real_A = torch.Tensor(opt.batchSize, input_nc, opt.fineSize, opt.fineSize) 
--  a tensor for 'batchSize' number of images which are the real structured images.
local real_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
--  a tensor for 'batchSize' number of output images which aims to produce image like real_B using real_A
local fake_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
-- this one stores both real_A and real_B along the dimension 2
local real_AB = torch.Tensor(opt.batchSize, output_nc + input_nc*opt.condition_GAN, opt.fineSize, opt.fineSize)
local fake_AB = torch.Tensor(opt.batchSize, output_nc + input_nc*opt.condition_GAN, opt.fineSize, opt.fineSize)
local errD, errG, errL1 = 0, 0, 0
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------

if opt.gpu > 0 then
   print('transferring to gpu...')
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   real_A = real_A:cuda();
   real_B = real_B:cuda(); fake_B = fake_B:cuda();
   real_AB = real_AB:cuda(); fake_AB = fake_AB:cuda();
   if opt.cudnn==1 then
      netG = util.cudnn(netG); netD = util.cudnn(netD);
   end
   netD:cuda(); netG:cuda(); criterion:cuda(); criterionAE:cuda();
   print('done')
else
	print('running model on CPU')
end

-- gradParameters are the stored (not calculated) values of the gradient and parameters are the values of weights and biases.
local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()



if opt.display then disp = require 'display' end



function createRealFake()
    -- load real
    -- time for loading data reset and started
    data_tm:reset(); data_tm:resume()
    local real_data, data_path = data:getBatch()
    data_tm:stop()
    -- time stopped for data loading
    
    real_A:copy(real_data[{ {}, idx_A, {}, {} }])
    real_B:copy(real_data[{ {}, idx_B, {}, {} }])
    -- real_A and real_B are added along dim 2
    if opt.condition_GAN==1 then
        real_AB = torch.cat(real_A,real_B,2)
    else
        real_AB = real_B -- unconditional GAN, only penalizes structure in B
    end
    -- mu = torch.zeros(256)
    -- sigma = 10000*torch.eye(256)
    
    -- real_A = torch.zeros(1,3,256,256)
    n = torch.zeros(1,1,16,16)
    local y = 1
    local z = 1
    while y<17 do
    	z=1
      while z<17 do 
	      n[{1,1,z,y}] = torch.uniform()*256
	      -- n[{1,2,z,y}] = torch.uniform()*256
	      -- n[{1,3,z,y}] = torch.uniform()*256
	      z = z+1
	     end 
      y = y + 1
    end
    -- n[{1,1,1,1}] = torch.uniform()*256
    if opt.gpu>0 then 
      n = n:cuda();
    end
      fake_B = netG:forward({n, real_A})
    -- print(sample:size())
    -- created fake using real_A as input
    -- print(fake_B:size())
    graph.dot(netG.fg, 'Noise_16', 'Noise_16')
    -- print(netG:get(8).output)
    -- if (counter==0)  then
    --   gen_out_old = netG:get(80).output
    -- -- optim.adam(fGx, parametersG, optimStateG)
    -- else
    --   print(netG:get(80).output-gen_out_old)
    --   gen_out_old = netG:get(80).output
    -- end
    -- k = netG:forward({n,real_A})
    if opt.condition_GAN==1 then
        fake_AB = torch.cat(real_A,fake_B,2)
    else
        fake_AB = fake_B -- unconditional GAN, only penalizes structure in B
    end
    local predict_real = netD:forward(real_AB)  
    local predict_fake = netD:forward(fake_AB)
end

-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
    netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    
    gradParametersD:zero()
    
    -- Real
    local output = netD:forward(real_AB)
    local label = torch.FloatTensor(output:size()):fill(real_label)
    if opt.gpu>0 then 
    	label = label:cuda()
    end
    -- Caclulate the error for misclassifying the real data points as fake
    local errD_real = criterion:forward(output, label)
    -- finds gradient wrt the output of the net only
    local df_do = criterion:backward(output, label) -- 1X1X30X30 = size of dicriminator 
    -- print(df_do:size())
    -- backpropogate 
    netD:backward(real_AB, df_do)
    
    -- Caclulate the error for misclassifying the fake data points as real
    local output = netD:forward(fake_AB)
    label:fill(fake_label)
    local errD_fake = criterion:forward(output, label)
    local df_do = criterion:backward(output, label)
    netD:backward(fake_AB, df_do)
    
    errD = (errD_real + errD_fake)/2
    
    return errD, gradParametersD
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
    netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    
    gradParametersG:zero()
    
    -- GAN loss
    local df_dg = torch.zeros(fake_B:size())
    if opt.gpu>0 then 
    	df_dg = df_dg:cuda();
    end
    
    if opt.use_GAN==1 then
       local output = netD.output -- netD:forward{input_A,input_B} was already executed in fDx, so save computation
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
    local df_do_AE = torch.zeros(fake_B:size())
    if opt.gpu>0 then 
    	df_do_AE = df_do_AE:cuda();
    end
    if opt.use_L1==1 then
       errL1 = criterionAE:forward(fake_B, real_B)
       df_do_AE = criterionAE:backward(fake_B, real_B)
    else
        errL1 = 0
    end
    
    netG:backward(real_A, df_dg + df_do_AE:mul(opt.lambda))
    
    return errG, gradParametersG
end




-- train
local best_err = nil
paths.mkdir(opt.checkpoints_dir)
paths.mkdir(opt.checkpoints_dir .. '/' .. opt.name)

-- save opt
file = torch.DiskFile(paths.concat(opt.checkpoints_dir, opt.name, 'opt.txt'), 'w')
file:writeObject(opt)
file:close()
local counter = 0
for epoch = 1, opt.niter do
    epoch_tm:reset()
    for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
        tm:reset()
        
        -- load a batch and run G on that batch
        createRealFake()
        -- (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        if opt.use_GAN==1 then optim.adam(fDx, gradParametersD, optimStateD) end
        
        -- (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        optim.adam(fGx, parametersG, optimStateG)
        
        -- display
        
        
        counter = counter + 1
        if counter % opt.display_freq == 0 and opt.display then
            createRealFake()
            if opt.preprocess == 'colorization' then 
                local real_A_s = util.scaleBatch(real_A:float(),100,100)
                local fake_B_s = util.scaleBatch(fake_B:float(),100,100)
                local real_B_s = util.scaleBatch(real_B:float(),100,100)
                disp.image(util.deprocessL_batch(real_A_s), {win=opt.display_id, title=opt.name .. ' input'})
                disp.image(util.deprocessLAB_batch(real_A_s, fake_B_s), {win=opt.display_id+1, title=opt.name .. ' output'})
                disp.image(util.deprocessLAB_batch(real_A_s, real_B_s), {win=opt.display_id+2, title=opt.name .. ' target'})
            else 
            	  -- print(netG:get(8).output)
				        -- if (counter/opt.display_freq==1)  then
				        --   gen_out_old = netG:get(1).weight
				        --   -- print("YESSSSSS")
				        -- -- optim.adam(fGx, parametersG, optimStateG)
				        -- else
                n = torch.zeros(1,1,16,16)
                local y = 1
                local z = 1
                while y<17 do
                  z=1
                  while z<17 do 
                    n[{1,1,z,y}] = torch.uniform()*256
                    -- n[{1,2,z,y}] = torch.uniform()*256
                    -- n[{1,3,z,y}] = torch.uniform()*256
                    z = z+1
                   end 
                  y = y + 1
                end
                -- n[{1,1,1,1}] = torch.uniform()*256
                if opt.gpu>0 then 
                  n = n:cuda();
                end
				        if (counter/opt.display_freq==1)  then
                  print("Hello from the inside")
                  Generator_out = netG:forward({n, real_A})
                  Generator_out_f = Generator_out:float()
                  netOut = netG:get(64).output
                  gen_out_old = Generator_out_f
                  noise_out = util.normalize(netOut:float())
                  noise_out_old = noise_out
                -- optim.adam(fGx, parametersG, optimStateG)
                else
                  Generator_out = netG:forward({n, real_A})
                  Generator_out_f = Generator_out:float()
                  netOut = netG:get(64).output      
                  noise_out = util.normalize(netOut:float())
                  a = noise_out-noise_out_old
                  b = Generator_out_f-gen_out_old
                  disp.image(util.scaleBatch(a,100,100),{win=opt.display_id, title=opt.name .. ' noise difference'})
                  disp.image(util.scaleBatch(b,100,100),{win=opt.display_id+1, title=opt.name .. ' Generator difference'})
			            disp.image(util.scaleBatch(noise_out:float(),100,100),{win=opt.display_id+2, title=opt.name .. 'difference'})
                  noise_out_old = noise_out
                  gen_out_old = Generator_out_f
                end
                -- netOut = netG:get(63).output
                -- noise_out = util.normalize(netOut:float())
                -- print(noise_out:size())
				          -- gen_out_old = netG:get(1).weight
				        -- end  
				        -- fake_AB = torch.cat(input,Generator_out,2)
				        -- output = util.deprocess_batch(Generator_out)
				        -- input = util.deprocess_batch(input):float()
				        -- output = output:float()
				        -- target = util.deprocess_batch(target):float()
            	  disp.image(util.deprocess_batch(util.scaleBatch(real_A:float(),100,100)), {win=opt.display_id+3, title=opt.name .. ' input'})
                disp.image(util.deprocess_batch(util.scaleBatch(fake_B:float(),100,100)), {win=opt.display_id+4, title=opt.name .. ' output'})
                disp.image(util.deprocess_batch(util.scaleBatch(real_B:float(),100,100)), {win=opt.display_id+5, title=opt.name .. ' target'})
            end
        end
      
        -- write display visualization to disk
        --  runs on the first batchSize images in the opt.phase set
        if counter % opt.save_display_freq == 0 and opt.display then
            local serial_batches=opt.serial_batches
            opt.serial_batches=1
            opt.serial_batch_iter=1
            
            local image_out = nil
            local N_save_display = 10
            for i3=1, torch.floor(N_save_display/opt.batchSize) do
            
                createRealFake()
                print('save to the disk')
                if opt.preprocess == 'colorization' then 
                    for i2=1, fake_B:size(1) do
                        if image_out==nil then image_out = torch.cat(util.deprocessL(real_A[i2]:float()),util.deprocessLAB(real_A[i2]:float(), fake_B[i2]:float()),3)/255.0
                        else image_out = torch.cat(image_out, torch.cat(util.deprocessL(real_A[i2]:float()),util.deprocessLAB(real_A[i2]:float(), fake_B[i2]:float()),3)/255.0, 2) end
                    end
                else
                    for i2=1, fake_B:size(1) do
                        if image_out==nil then image_out = torch.cat(util.deprocess(real_A[i2]:float()),util.deprocess(fake_B[i2]:float()),3)
                        else image_out = torch.cat(image_out, torch.cat(util.deprocess(real_A[i2]:float()),util.deprocess(fake_B[i2]:float()),3), 2) end
                    end
                end
            end
            image.save(paths.concat(opt.checkpoints_dir,  opt.name , counter .. '_train_res.png'), image_out)
            
            opt.serial_batches=serial_batches
        end
        
        -- logging
        if counter % opt.print_freq == 0 then
            print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                    .. '  Err_G: %.4f  Err_D: %.4f  ErrL1: %.4f'):format(
                     epoch, ((i-1) / opt.batchSize),
                     math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                     tm:time().real / opt.batchSize, data_tm:time().real / opt.batchSize,
                     errG and errG or -1, errD and errD or -1, errL1 and errL1 or -1))
        end
        
        -- save latest model
        if counter % opt.save_latest_freq == 0 then
            print(('saving the latest model (epoch %d, iters %d)'):format(epoch, counter))
            torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_G.t7'), netG:clearState())
            torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_D.t7'), netD:clearState())
        end
        
    end
    
    
    parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
    parametersG, gradParametersG = nil, nil
    
    if epoch % opt.save_epoch_freq == 0 then
        torch.save(paths.concat(opt.checkpoints_dir, opt.name,  epoch .. '_net_G.t7'), netG:clearState())
        torch.save(paths.concat(opt.checkpoints_dir, opt.name, epoch .. '_net_D.t7'), netD:clearState())
    end
    
    print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
    parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
    parametersG, gradParametersG = netG:getParameters()


end