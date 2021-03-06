require 'rnn'
require 'DataLoader'
require 'utils'


local Focus2BiLSTM = torch.class('Focus2BiLSTM')

function Focus2BiLSTM:__init(opt)
  -- set CudaTensor
  if opt.useGPU > 0 then
    require 'cunn'
    require 'cutorch'
    cutorch.setDevice(2)
    torch.setdefaulttensortype('torch.CudaTensor')
  end
  
  -- create a output folder
  os.execute("mkdir " .. opt.modelDirectory)
  self.modelDirectory = opt.modelDirectory
  
  -- load data
  self.dataLoader = DataLoader(opt.trainDataFile, opt.batchSize)
  self.validDataLoader = DataLoader(opt.validDataFile, 1)

  -- setting
  self.printEveryNumIter = opt.printEveryNumIter
  self.maxEpochs = opt.maxEpochs
  -- parameters
  self.hiddenSize = opt.hiddenSize
  self.nIndex = math.max(self.dataLoader.maxIndex, self.validDataLoader.maxIndex)
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
        wordEmbed.weight[i][j - 1] = tonumber(line_tokens[j])
      end
    end
    wordEmbedFile:close()
  end

  local lstm_1_fwd = nn.Sequential()
    :add(wordEmbed)
    :add(nn.LSTM(self.hiddenSize, self.hiddenSize))
  local lstm_1_bwd = nn.Sequential()
    :add(wordEmbed)
    :add(nn.LSTM(self.hiddenSize, self.hiddenSize))
  local fwdSeq_1 = nn.Sequencer(lstm_1_fwd)
  local bwdSeq_1 = nn.Sequencer(lstm_1_bwd)
  local backward_1 = nn.Sequential()
    :add(nn.ReverseTable())
    :add(bwdSeq_1)
    :add(nn.ReverseTable())
  local concat_1 = nn.ConcatTable()
    :add(fwdSeq_1):add(backward_1)

  local lstm_2 = nn.Sequential()
    :add(nn.LSTM(2 * self.hiddenSize, self.hiddenSize))
  local fwdSeq_2 = nn.Sequencer(lstm_2:clone())
  local bwdSeq_2 = nn.Sequencer(lstm_2:clone())
  local backward_2 = nn.Sequential()
    :add(nn.ReverseTable())
    :add(bwdSeq_2)
    :add(nn.ReverseTable())
  local concat_2 = nn.ConcatTable()
    :add(fwdSeq_2):add(backward_2)

  local brnn = nn.Sequential()
    :add(concat_1)
    :add(nn.ZipTable())
    :add(nn.Sequencer(nn.JoinTable(1, 1)))
    :add(concat_2)
    :add(nn.ZipTable())
    :add(nn.Sequencer(nn.JoinTable(1, 1)))
    :add(nn.Sequencer(nn.Linear(self.hiddenSize * 2, self.nClass)))
    :add(nn.Sequencer(nn.LogSoftMax()))
  self.biLSTM = cudacheck(brnn)
  
  -- criterion
  self.criterion = cudacheck(nn.SequencerCriterion(nn.ClassNLLCriterion()))
end

function Focus2BiLSTM:train()
  local epochLoss, accumLoss = 0, 0
  local maxIter = self.dataLoader.numBatch * self.maxEpochs
  for i = 1, maxIter do
    xlua.progress(i, maxIter)
    local inputs, targets = unpack(self.dataLoader:nextBatch())

    self.biLSTM:zeroGradParameters()
    
    local inputSeq = inputs:split(1, 1)
    for j = 1, #inputSeq, 1 do
      inputSeq[j] = torch.squeeze(inputSeq[j])
    end
    local targetSeq = targets:split(1, 1)
    for j = 1, #targetSeq, 1 do
      targetSeq[j] = torch.squeeze(targetSeq[j])
    end
    local outputs = self.biLSTM:forward(inputSeq)
    local err = self.criterion:forward(outputs, targetSeq)
    
    epochLoss = epochLoss + err / inputs:size(1) / inputs:size(2)
    accumLoss = accumLoss + err / inputs:size(1) / inputs:size(2)
    
    local gradOutputs = self.criterion:backward(outputs, targetSeq)
    local gradInputs = self.biLSTM:backward(inputSeq, gradOutputs)
    
    self.biLSTM:updateParameters(0.1)

    if i % self.printEveryNumIter == 0 then
      print(string.format("[Iter %d]: %f", i, accumLoss / self.printEveryNumIter))
      accumLoss = 0
    end
    
    -- evaluate and save model
    if i % (10 * self.dataLoader.numBatch) == 0 then
      local epoch = i / self.dataLoader.numBatch
      print(string.format("[Epoch %d]: %f", epoch, epochLoss / self.dataLoader.numBatch))
      self:evaluate()
      torch.save(self.modelDirectory.."/biLSTM_"..epoch, self.biLSTM)
      epochLoss = 0
    end
  end
end

function Focus2BiLSTM:evaluate()
  local count = 0
  for i = 1, self.validDataLoader.dataSize, 1 do
    local inputs, targets = unpack(self.validDataLoader:nextBatch())
    self.biLSTM:zeroGradParameters()
    
    local inputSeq = inputs:split(1, 1)
    for j = 1, #inputSeq, 1 do
      inputSeq[j] = torch.LongTensor(1):fill(torch.squeeze(inputSeq[j]))
    end
    local outputs = self.biLSTM:forward(inputSeq)
    local _, idx = torch.max(torch.concat(outputs, 1), 2)
    if torch.all(torch.eq(torch.squeeze(targets), torch.squeeze(cudacheck(idx)))) then
      count = count + 1
    end
  end
  print("count: "..count)
end
