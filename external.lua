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
	protos.cnn:evaluate()
	protos.lm:evaluate()
	loader:resetIterator(split)
	images = net_utils.prepro(images, false, opt.gpuid >= 0)
	local feats = protos.cnn:forward(images)
	local expanded_feats = protos.expander:forward(feats)
	local logprobs = protos.lm:forward{expanded_feats, labels}
	return protos.crit:forward(logprobs, labels) -- return loss
end

return ntalk

