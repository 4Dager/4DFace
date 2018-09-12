require 'torch'
require 'nn'
require 'dpnn'
require 'image'

io.stdout:setvbuf 'no'
torch.setdefaulttensortype('torch.FloatTensor')

torch.setnumthreads(1)

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Face recognition server.')
cmd:text()
cmd:text('Options:')

cmd:option('-model', './models/fcn.t7', 'Path to model.')
cmd:option('-imgDim', 96, 'Image dimension. nn1=224, nn4=96')
cmd:option('-cuda', false)
cmd:text()

opt = cmd:parse(arg or {})


net = torch.load(opt.model)
net:evaluate()


local imgCuda = nil
if opt.cuda then
   require 'cutorch'
   require 'cunn'
   net = net:cuda()
   imgCuda = torch.CudaTensor(1, 3, opt.imgDim, opt.imgDim)
end

local img = torch.Tensor(1, 3, opt.imgDim, opt.imgDim)
while true do
   local imgPath = io.read("*line")
   if imgPath and imgPath:len() ~= 0 then
      img[1] = image.load(imgPath, 3, 'float')
      img[1] = image.scale(img[1], opt.imgDim, opt.imgDim)
      local rep
      if opt.cuda then
         imgCuda:copy(img)
         rep = net:forward(imgCuda):float()
      else
         rep = net:forward(img)
      end
      local sz = rep:size(1)
      for i = 1,sz do
         io.write(rep[i])
         if i < sz then
            io.write(',')
         end
      end
      io.write('\n')
      io.stdout:flush()
   end
end
