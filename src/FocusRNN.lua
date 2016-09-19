require 'rnn'
require 'DataLoader'
require 'utils'


local FocusRNN = torch.class('FocusRNN')

function FocusRNN:__init(opt)
  -- set CudaTensor
  if opt.useGPU > 0 then
    require 'cunn'
    require 'cutorch'
    torch.setdefaulttensortype('torch.CudaTensor')
  end
  
  -- load data
  self.dataLoader = DataLoader(opt.trainDataFile, opt.batchSize)
  self.validDataLoader = DataLoader(opt.validDataFile, 1)
  
  -- setting
  self.printEveryNumIter = opt.printEveryNumIter
  self.maxEpochs = opt.maxEpochs
  -- parameters
  self.hiddenSize = opt.embedSize
  self.nIndex = self.dataLoader.maxIndex
  self.nClass = self.dataLoader.maxClass
  
  -- model
  local wordEmbed = nn.LookupTable(self.nIndex, self.hiddenSize)
  if opt.useEmbed > 0 then
    -- load Word Embedding
    local wordEmbedFile = io.open(opt.wordEmbeddingFile, 'r')
    local tokens = split(wordEmbedFile:read(), " ")
    local vocabSize, dimension = tonumber(tokens[1]), tonumber(tokens[2])
    for i = 1, vocabSize, 1 do
      -- xlua.progress(i, vocabSize)
      local line_tokens = split(wordEmbedFile:read(), " ")
      for j = 2, #line_tokens, 1 do
        wordEmbed.weight[i + 1][j - 1] = tonumber(line_tokens[j])
      end
    end
  end

  local r = nn.Recurrent(
    self.hiddenSize, wordEmbed,
    nn.Linear(self.hiddenSize, self.hiddenSize), nn.Sigmoid()
  )
  local rnn = nn.Sequential()
    :add(r)
    :add(nn.Linear(self.hiddenSize, self.nClass))
    :add(nn.LogSoftMax())
  self.rnn = cudacheck(nn.Recursor(rnn))
  
  -- criterion
  self.criterion = cudacheck(nn.ClassNLLCriterion())
end

function FocusRNN:train()
  local epochLoss, accumLoss = 0, 0
  local maxIter = self.dataLoader.numBatch * self.maxEpochs
  for i = 1, maxIter do
    local data, mark = unpack(self.dataLoader:nextBatch())
    self.rnn:zeroGradParameters()
    self.rnn:forget()
    -- forward
    local stepOutputs, err = {}, 0
    for step = 1, data:size(1), 1 do
      stepOutputs[step] = self.rnn:forward(data[step])
      err = err + self.criterion:forward(stepOutputs[step], mark[step])
    end
    -- error
    epochLoss = epochLoss + err / data:size(1) / data:size(2)
    accumLoss = accumLoss + err / data:size(1) / data:size(2)
    -- backward
    local gradStepOutputs, gradStepInputs = {}, {}
    for step = data:size(1), 1, -1 do
      gradStepOutputs[step] = self.criterion:backward(stepOutputs[step], mark[step])
      gradStepInputs[step] = self.rnn:backward(data[step], gradStepOutputs[step])
    end
    -- update
    self.rnn:updateParameters(0.01)
    -- print
    if i % self.printEveryNumIter == 0 then
      print(string.format("[Iter %d]: %f", i, accumLoss / self.printEveryNumIter))
      accumLoss = 0
      -- test now
      -- self:evaluate()
    end
    if i % self.dataLoader.numBatch == 0 then
      local epoch = i / self.dataLoader.numBatch
      print(string.format("[Epoch %d]: %f", epoch, epochLoss / self.dataLoader.numBatch))
      -- test now
      self:evaluate()
      epochLoss = 0
    end
  end
end

function FocusRNN:evaluate()
  local count = 0
  for i = 1, self.validDataLoader.dataSize, 1 do
    local data, mark = unpack(self.validDataLoader:nextBatch())
    self.rnn:zeroGradParameters()
    self.rnn:forget()
    local outputs, indexOutputs = {}, {}
    for step = 1, data:size(1), 1 do
      outputs[step] = self.rnn:forward(data[step])
      _, idx = torch.max(outputs[step], 2)
      indexOutputs[#indexOutputs + 1] = idx[1][1]
    end
    print(torch.Tensor(indexOutputs))
    print(mark:t()[1])
    if torch.all(torch.eq(torch.Tensor(indexOutputs),  mark:t()[1])) then
      count = count + 1
    end
  end
  print("count: "..count)
end

