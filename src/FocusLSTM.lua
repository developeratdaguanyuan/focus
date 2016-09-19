require 'rnn'
require 'DataLoader'
require 'utils'


local FocusLSTM = torch.class('FocusLSTM')

function FocusLSTM:__init(opt)
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
  self.hiddenSize = opt.hiddenSize
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
  local rnn = nn.Sequential()
    :add(wordEmbed)
    :add(nn.LSTM(self.hiddenSize, self.hiddenSize))
    :add(nn.Linear(self.hiddenSize, self.nClass))
    :add(nn.LogSoftMax())

  self.rnn = cudacheck(nn.Sequencer(rnn))
  
  -- criterion
  self.criterion = cudacheck(nn.SequencerCriterion(nn.ClassNLLCriterion()))
end

function FocusLSTM:train()
  local epochLoss, accumLoss = 0, 0
  local maxIter = self.dataLoader.numBatch * self.maxEpochs
  for i = 1, maxIter do
    local inputs, targets = unpack(self.dataLoader:nextBatch())

    self.rnn:zeroGradParameters()

    local outputs = self.rnn:forward(inputs)
    local err = self.criterion:forward(outputs, targets)

    epochLoss = epochLoss + err / inputs:size(1) / inputs:size(2)
    accumLoss = accumLoss + err / inputs:size(1) / inputs:size(2)

    local gradOutputs = self.criterion:backward(outputs, targets)
    local gradInputs = self.rnn:backward(inputs, gradOutputs)

    self.rnn:updateParameters(0.1)
    -- print
    if i % self.printEveryNumIter == 0 then
      print(string.format("[Iter %d]: %f", i, accumLoss / self.printEveryNumIter))
      accumLoss = 0
    end
    if i % self.dataLoader.numBatch == 0 then
      local epoch = i / self.dataLoader.numBatch
      print(string.format("[Epoch %d]: %f", epoch, epochLoss / self.dataLoader.numBatch))
      self:evaluate()
      epochLoss = 0
    end
  end
end

function FocusLSTM:evaluate()
  local count = 0
  for i = 1, self.validDataLoader.dataSize, 1 do
    local inputs, targets = unpack(self.validDataLoader:nextBatch())
    self.rnn:zeroGradParameters()
    local outputs = self.rnn:forward(inputs)
    local _, idx = torch.max(outputs, 3)

    if torch.all(torch.eq(torch.squeeze(targets),  torch.squeeze(idx))) then
      count = count + 1
    end
  end
  print("count: "..count)
end

