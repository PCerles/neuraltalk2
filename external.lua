-- exotics
require 'loadcaffe'
-- local imports
local utils = require 'misc.utils'
require 'misc.DataLoader'
require 'misc.DataLoaderRaw'
require 'misc.LanguageModel'
require 'cutorch'
--require 'cudnn'


local net_utils = require 'misc.net_utils'
local ntalk = {}

function ntalk.getProtos(gpu_id) -- default  -1
	local model =  'model/d1-501-1448236541.t7_cpu.t7' -- could replace with input
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

function ntalk.getLoss(protos, images, captions)
	
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

