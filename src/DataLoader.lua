require 'utils'


local DataLoader = torch.class('DataLoader')

function DataLoader:__init(dataPath, batchSize)
  self.batchSize = batchSize == nil and 10 or batchSize
  self.data, self.mark, self.maxIndex, self.maxClass = unpack(self:createData(dataPath))
  self.dataSize = #self.data
  self.numBatch = math.floor(self.dataSize/self.batchSize)
  self.headList, self.cntList = unpack(self:groupData())
end

function DataLoader:nextBatch()
  if self.currentIndex == nil or self.currentIndex + self.batchSize > self.dataSize then
    self.currentIndex = 1
    self.indices = self:rerank(self.headList, self.cntList, self.dataSize)
  end

  -- select data indices
  local dataIndex = self.indices:narrow(1, self.currentIndex, self.batchSize)
  self.currentIndex = self.currentIndex + self.batchSize

  -- construct current batch data
  local maxStep = self.data[dataIndex[1]]:size(1)
  local currentDataBatch = torch.LongTensor(self.batchSize, maxStep):fill(1)
  local currentMarkBatch = torch.LongTensor(self.batchSize, maxStep):fill(1)
  for i = 1, self.batchSize, 1 do
    currentDataBatch[{{i}, {maxStep - self.data[dataIndex[i]]:size(1) + 1, maxStep}}]
      = self.data[dataIndex[i]]
    currentMarkBatch[{{i}, {maxStep - self.data[dataIndex[i]]:size(1) + 1, maxStep}}]
      = self.mark[dataIndex[i]]
  end

  return {cudacheck(currentDataBatch:t()), cudacheck(currentMarkBatch:t())}
end

function DataLoader:rerank(headList, cntList, dataSize)
  local list = torch.LongTensor(dataSize):zero()
  for i = 1, #headList, 1 do
    list[{{headList[i], headList[i] + cntList[i] - 1}}] = torch.LongTensor.torch.randperm(cntList[i]):add(headList[i] - 1)
  end
  return list
end

function DataLoader:createData(path)
  local file = io.open(path, 'r')
  local dataset = {}
  local markset = {}
  local maxIndex, maxClass = 0, 0
  while true do
    local line = file:read()
    if line == nil then
      break
    end
    local tokens = split(line, "\t")
    local data = torch.LongTensor(split(tokens[1], ","))
    local mark = torch.LongTensor(split(tokens[2], ","))
    dataset[#dataset + 1] = data
    markset[#markset + 1] = mark

    maxIndex = math.max(maxIndex, torch.max(data))
    maxClass = math.max(maxClass, torch.max(mark))
  end
  return {dataset, markset, maxIndex, maxClass}
end

function DataLoader:groupData()
  local headList = {}
  local cntList = {}

  local head, cnt = 1, 1
  for i = 2, #self.data, 1 do
    if self.data[i]:size(1) == self.data[head]:size(1) then
      cnt = cnt + 1
    else
      headList[#headList + 1] = head
      cntList[#cntList + 1] = cnt
      head = i
      cnt = 1
    end
  end
  headList[#headList + 1] = head
  cntList[#cntList + 1] = cnt

  return {headList, cntList}
end

