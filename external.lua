-- other requirements are required in main.lua
-- exotics
require 'loadcaffe'
-- local imports
require 'cutorch'
require 'misc.LanguageModel'

local net_utils = require 'misc.net_utils'
local ntalk = {}


-- Returns the neuraltalk2 protos given model path and gpuid
function ntalk.getProtos(model, gpuid) -- default  -1
	local seed = 123
	local checkpoint = torch.load(model)
	-- Load the networks from model checkpoint 
	local protos = checkpoint.protos
	protos.expander = nn.FeatExpander(checkpoint.opt['seq_per_img'])
	protos.crit = nn.LanguageModelCriterion()
	protos.lm:createClones() -- reconstruct clones inside the language model
	if gpuid >= 0 then for k,v in pairs(protos) do v:cuda() end end
	return protos
end

-- Returns the loss between the predicted caption based on the
-- image and the real caption given in labels
function ntalk.getLoss(protos, images, labels)
	local batch_size = 32
	protos.cnn:evaluate()
	protos.lm:evaluate()
	local images_new = torch.ByteTensor(batch_size, 3, 256, 256)
	for i = 1,batch_size do
		local im = images[i]
		local yByte = torch.ByteTensor(im:size()):copy(im)
		local im2 = image.scale(yByte, 256, 256)
		images_new[i] = im2 
	end
	
	images_new = net_utils.prepro(images_new, false, true)
	freeMemory, totalMemory = cutorch.getMemoryUsage(1)
	local feats = protos.cnn:forward(images_new)
	local expanded_feats = protos.expander:forward(feats)
	print(expanded_feats)
	print(labels)
	local logprobs = protos.lm:forward({expanded_feats, labels})
	return protos.crit:forward(logprobs, labels) -- return loss
end

return ntalk

