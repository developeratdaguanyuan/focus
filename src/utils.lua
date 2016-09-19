-- split
function split(line, sep)
  if sep == nil then
    sep = "%s"
  end
  local t = {}; local i = 1
  for word in string.gmatch(line, "([^"..sep.."]+)") do
    t[i] = word; i = i + 1
  end
  return t
end

-- cuda
function cudacheck(input)
    if torch.Tensor():type() == 'torch.CudaTensor' then
        input = input:cuda()
    end 
    return input
end

-- clone modules with shared parameters
function cloneModulesWithSharedParameters(m, t)
  local clones = {}
  local params, gradParams = m:parameters()
  local mem = torch.MemoryFile("w"):binary()
  mem:writeObject(m)
  for _t = 1, t do
    local reader = torch.MemoryFile(mem:storage(), "r"):binary()
    local clone = reader:readObject()
    reader:close()
    local cloneParams, cloneGradParams = clone:parameters()
    for i = 1, #params do
      cloneParams[i]:set(params[i])
      cloneGradParams[i]:set(gradParams[i])
    end
    clones[_t] = clone
    collectgarbage()
  end
  mem:close()
  return clones
end

-- main()
function main()
  local m = nn.LookupTable(5, 5)
  local m1, m2 = table.unpack(cloneModulesWithSharedParameters(m, 2))

  local m11, m12 = m1:parameters()
  local m21, m22 = m1:parameters()

  m11[1][1][2] = 888
  print(m11[1])
  print(m21[1])
end
