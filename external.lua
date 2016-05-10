-- other requirements are required in main.lua
-- exotics
require 'loadcaffe'
-- local imports
require 'cutorch'
require 'misc.LanguageModel'
require 'image'

local net_utils = require 'misc.net_utils'
local ntalk = {}


-- Returns the neuraltalk2 protos given model path and gpuid
function ntalk.getProtos(model, gpuid) -- default  -1
	local seed = 123
	local checkpoint = torch.load(model)
	-- Load the networks from model checkpoint 
	local protos = checkpoint.protos

	net_utils.unsanitize_gradients(protos.cnn)
	local lm_modules = protos.lm:getModulesList()
	for k,v in pairs(lm_modules) do net_utils.unsanitize_gradients(v) end

	protos.expander = nn.FeatExpander(checkpoint.opt['seq_per_img']) -- not in checkpoints, create manually
	protos.crit = nn.LanguageModelCriterion()
	--protos.lm:createClones() -- reconstruct clones inside the language model
	if gpuid >= 0 then for k,v in pairs(protos) do v:cuda() end end

	protos.lm:createClones()
	collectgarbage()
	return protos
end

-- Returns the loss between the predicted caption based on the
-- image and the real caption given in labels
-- labels size 16 x 32 
-- 
function ntalk.getLoss(protos, images, labels)
	local batch_size = 64
	local seq_per_img = 5
	local seq_length = 16

	-- repeat, reshape labels
	labels = torch.repeatTensor(labels, 1, seq_per_img)  
	labels = torch.reshape(labels, batch_size * seq_per_img, seq_length)
	labels = labels:transpose(1,2):contiguous()
	-- to mirror neuraltalk2 training
	protos.cnn:training()
	protos.lm:training()

	protos.cnn:zeroGradParameters()
	protos.lm:zeroGradParameters()
	-- end of setting nets up 

	-- upsample images to 256x256 and put them in big tensor
	local images_new = torch.ByteTensor(batch_size, 3, 256, 256)
	local im2, yByte
	for i = 1,batch_size do
		yByte = torch.ByteTensor(images[i]:size()):copy(images[i])
		images_new[i] = image.scale(yByte, 256, 256) 
	end
	-- end of image manicuring

	images_new = net_utils.prepro(images_new, false, true)
	--images_new:cuda()
	-- forward pass
	local feats = protos.cnn:forward(images_new)
	local expanded_feats = protos.expander:forward(feats)
	local logprobs = protos.lm:forward({expanded_feats, labels})

	local errN = protos.crit:forward(logprobs, labels) -- return loss
	dlogprobs = protos.crit:updateGradInput(logprobs, labels)
	
	local dexpanded_feats, ddummy = unpack(protos.lm:updateGradInput({expanded_feats, labels}, dlogprobs))
	local dfeats = protos.expander:updateGradInput(feats, dexpanded_feats)
	local dx = protos.cnn:updateGradInput(images_new, dfeats)

	local im = torch.DoubleTensor(dx:size()):copy(dx)
	local im_new = torch.DoubleTensor(batch_size, 3, 64, 64)
	local rescaled
	for i = 1,batch_size do
		rescaled = image.scale(im[i], 64, 64)
		im_new[i] = rescaled
	end	
	return errN,im_new
end

return ntalk

