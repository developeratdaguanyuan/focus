require 'FocusRNN'
require 'FocusLSTM'
require 'FocusBiLSTM'
require 'Focus2BiLSTM'


local cmd = torch.CmdLine()
cmd:option('-trainDataFile', '../data/sentence_focus_train.txt', 'training data file')
cmd:option('-validDataFile', '../data/sentence_focus_valid.txt', 'validation data file')
cmd:option('-modelDirectory', "../model")
cmd:option('-wordEmbeddingFile', '../data/embedding.txt')
cmd:option('-useEmbed', 1, 'whether to use word embedding')
cmd:option('-embedSize', 300, 'embedding size')
cmd:option('-hiddenSize', 300, 'hidden size')
cmd:option('-maxEpochs', 100, 'number of full passes through training data')
cmd:option('-batchSize', 100, 'number of data in a batch')
cmd:option('-useGPU', 1, 'whether to use GPU')
cmd:option('-printEveryNumIter', 10, 'print training loss every several iterations')

local opt = cmd:parse(arg)
--[[
local focusRNN = FocusRNN(opt);
focusRNN:train()]]
--[[
local focusLSTM = FocusLSTM(opt);
focusLSTM:train()]]
--[[
local focusBiLSTM = FocusBiLSTM(opt);
focusBiLSTM:train()]]
local focus2BiLSTM = Focus2BiLSTM(opt);
focus2BiLSTM:train()
