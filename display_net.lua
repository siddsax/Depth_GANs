require 'distributions'
require 'nngraph'
require 'image'
require 'nn.THNN'
mu = torch.zeros(100)
sigma = 100*torch.eye(100)
n = torch.zeros(1,3,100,100)
real_A = torch.zeros(1,3,256,256)

local i = 1
while i<101 do
	sample_1 = distributions.mvn.rnd(mu, sigma) -- a 
	sample_2 = distributions.mvn.rnd(mu, sigma) -- a 
	sample_3 = distributions.mvn.rnd(mu, sigma) -- a 
	n[{1,1,{1,100},i}] = sample_1
	n[{1,2,{1,100},i}] = sample_2
	n[{1,3,{1,100},i}] = sample_3
  i = i + 1
end
-- sample_1 = distributions.mvn.rnd(mu, sigma) -- a 
-- sample_2 = distributions.mvn.rnd(mu, sigma) -- a 
-- sample_3 = distributions.mvn.rnd(mu, sigma) -- a 
-- n[{1,1,{1,256},1}] = sample_1
-- n[{1,2,{1,256},1}] = sample_2
-- n[{1,3,{1,256},1}] = sample_3
-- input = torch.cat(real_A,n,4) -- 1X3X256X257
-- print(n)
input_nc = 3
ngf = 64
output_nc = 1

nngraph.setDebug(true)

-- local k = - nn.Select(4, 257)
-- local e0 = k - nn.SpatialConvolution(input_nc, input_nc, 2, 1, 1, 1, 0, 0)
-- -- input is (nc) x 256 x 256
-- local k1 = k - nn.Unsqueeze(4)
-- local e0 = - nn.SpatialConvolution(input_nc, input_nc, 2, 1, 1, 1, 0, 0)

-----------------------------------------------------------------------
-- input is (ngf) x 100 x 100
local n1 = - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
-- input is (ngf) x 50 x 50
local n2 = n1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
-- input is (ngf * 2) x 25 x 25
local n3 = n2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
-- input is (ngf * 4) x 12 x 12
local n4 = n3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
-- input is (ngf * 8) x 6 x 6
local n5 = n4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
-- input is (ngf * 8) x 3 x 3
local n6 = n5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)

-- input is (nc) x 256 x 256
local e1 = - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
-- input is (ngf) x 128 x 128
local e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
-- input is (ngf * 2) x 64 x 64
local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
-- input is (ngf * 4) x 32 x 32
local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
-- input is (ngf * 8) x 16 x 16
local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
-- input is (ngf * 8) x 8 x 8
local e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
-- input is (ngf * 8) x 4 x 4
local e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
-- input is (ngf * 8) x 2 x 2
local e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
-- input is (ngf * 8) x 1 x 1


local ne = {n6,e8} - nn.JoinTable(2)
-- local ne = {n6,e8} - nn.CAddTable()

local d1_ = ne - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8*2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
-- input is (ngf * 8) x 2 x 2
local d1 = {d1_,e7} - nn.JoinTable(2)
local d2_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
-- input is (ngf * 8) x 4 x 4
local d2 = {d2_,e6} - nn.JoinTable(2)
local d3_ = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
-- input is (ngf * 8) x 8 x 8
local d3 = {d3_,e5} - nn.JoinTable(2)
local d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
-- input is (ngf * 8) x 16 x 16
local d4 = {d4_,e4} - nn.JoinTable(2)
local d5_ = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
-- input is (ngf * 4) x 32 x 32
local d5 = {d5_,e3} - nn.JoinTable(2)
local d6_ = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
-- input is (ngf * 2) x 64 x 64
local d6 = {d6_,e2} - nn.JoinTable(2)
local d7_ = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
-- input is (ngf) x128 x 128
local d7 = {d7_,e1} - nn.JoinTable(2)
local d8 = d7 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, output_nc, 4, 4, 2, 2, 1, 1)
-- input is (nc) x 256 x 256

local o1 = d8 - nn.Tanh()
netG = nn.gModule({n1,e1},{o1})
graph.dot(netG.fg, 'graph', 'graph')

-- input is (ngf) x 128 x 128
-- local e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)

-- netG = nn.gModule(,{e1})
k = netG:forward({n,real_A})
print(k)

-- print(netG:forward(n))
--------------------------------------------
-- h1 = - nn.Linear(20,20)
-- h2 = - nn.Linear(10,10)
-- hh1 = h1 - nn.Tanh() - nn.Linear(20,1)
-- hh2 = h2 - nn.Tanh() - nn.Linear(10,1)
-- madd = {hh1,hh2} - nn.JoinTable(1)
-- -- madd = {hh1,hh2} - nn.CAddTable()

-- -- oA = madd - nn.Sigmoid()
-- -- oB = madd - nn.Tanh()
-- gmod = nn.gModule( {h1,h2}, {madd} )

-- x1 = torch.rand(20)
-- x2 = torch.rand(10)
-- print(gmod:forward({x1,x2}))

-- -- gmod:updateOutput({x1, x2})
-- print(gmod:forward({x1, x2}))
-- -- gmod:updateGradInput({x1, x2}, {torch.rand(1), torch.rand(1)})
-- -- graph.dot(netG.fg, 'netG', 'netG')