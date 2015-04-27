--
----  Copyright (c) 2014, Facebook, Inc.
----  All rights reserved.
----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree. 
----
ok,cunn = pcall(require, 'fbcunn')
if not ok then
    ok,cunn = pcall(require,'cunn')
    if ok then
        print("warning: fbcunn not found. Falling back to cunn") 
        LookupTable = nn.LookupTable
    else
        print("Could not find cunn or fbcunn. Either is required")
        os.exit()
    end
else
    deviceParams = cutorch.getDeviceProperties(1)
    cudaComputeCapability = deviceParams.major + deviceParams.minor/10
    LookupTable = nn.LookupTable
end
require('nngraph')
require('base')
stringx = require('pl.stringx')

cmd = torch.CmdLine()
cmd:option('-type','train','Executing Type, query or train')
cmd:option('-level','character','Word or character')
cmd:text()
cmd_param = cmd:parse(arg)
print(cmd_param.type,cmd_param.level)

function transfer_data(x)
  return x:cuda()
end

if cmd_param.level == 'character' then
  ptb = require('data_charLevel')
  err_factor = 5.6
  params = {    batch_size=20,
                seq_length=20,
                layers=3,
                decay=2,
                rnn_size=100,
                dropout=0.3,
                init_weight=0.1,
                lr=1,
                vocab_size=50,
                max_epoch=4,
                max_max_epoch=13,
                max_grad_norm=4}
  state_train = {data=transfer_data(ptb.traindataset(params.batch_size))}
  state_valid =  {data=transfer_data(ptb.validdataset(params.batch_size))}

  if cmd_param.type == 'query' and paths.filep('modelC_A4_temp.net') then
    A4 = torch.load('modelC_A4_temp.net')
    model = A4.model
    params = A4.params
  else 
    cmd_param.type = 'train'
    model = {}
  end
else
  ptb = require('data')
  err_factor = 1
  params = {batch_size=20,
                seq_length=20,
                layers=2,
                decay=2,
                rnn_size=200,
                dropout=0,
                init_weight=0.1,
                lr=1,
                vocab_size=10000,
                max_epoch=4,
                max_max_epoch=13,
                max_grad_norm=5}
  state_train = {data=transfer_data(ptb.traindataset(params.batch_size))}
  state_valid =  {data=transfer_data(ptb.validdataset(params.batch_size))}
  state_test =  {data=transfer_data(ptb.testdataset(params.batch_size))}

  if cmd_param.type == 'query' and paths.filep('model_A4_temp.net') then
    A4 = torch.load('model_A4_temp.net')
    model = A4.model
    params = A4.params
  else 
    cmd_param.type = 'train'
    model = {}
  end
end

-- Train 1 day and gives 82 perplexity.
--[[
local params = {batch_size=20,
                seq_length=35,
                layers=2,
                decay=1.15,
                rnn_size=1500,
                dropout=0.65,
                init_weight=0.04,
                lr=1,
                vocab_size=10000,
                max_epoch=14,
                max_max_epoch=55,
                max_grad_norm=10}
               ]]--

-- Trains 1h and gives test 115 perplexity.
--[[params = {batch_size=20,
                seq_length=20,
                layers=4,
                decay=2,
                rnn_size=200,
                dropout=0.3,
                init_weight=0.1,
                lr=1,
                vocab_size=10000,
                max_epoch=4,
                max_max_epoch=13,
                max_grad_norm=4}
--]]

--local state_train, state_valid, state_test

--local paramx, paramdx

function lstm(i, prev_c, prev_h)
  local function new_input_sum()
    local i2h            = nn.Linear(params.rnn_size, params.rnn_size)
    local h2h            = nn.Linear(params.rnn_size, params.rnn_size)
    return nn.CAddTable()({i2h(i), h2h(prev_h)})
  end
  local in_gate          = nn.Sigmoid()(new_input_sum())
  local forget_gate      = nn.Sigmoid()(new_input_sum())
  local in_gate2         = nn.Tanh()(new_input_sum())
  local next_c           = nn.CAddTable()({
    nn.CMulTable()({forget_gate, prev_c}),
    nn.CMulTable()({in_gate,     in_gate2})
  })
  local out_gate         = nn.Sigmoid()(new_input_sum())
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
  return next_c, next_h
end

function create_network()
  local x                = nn.Identity()()
  local y                = nn.Identity()()
  local prev_s           = nn.Identity()()
  local i                = {[0] = LookupTable(params.vocab_size , params.rnn_size)(x)}
  local next_s           = {}
  local split         = {prev_s:split(2 * params.layers)}
  for layer_idx = 1, params.layers do
    local prev_c         = split[2 * layer_idx - 1]
    local prev_h         = split[2 * layer_idx]
    local dropped        = nn.Dropout(params.dropout)(i[layer_idx - 1])
    local next_c, next_h = lstm(dropped, prev_c, prev_h)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    i[layer_idx] = next_h -- a params.rnn size vector , i[params.layers] is the final state of the given rnn[n]
  end 
  local h2y              = nn.Linear(params.rnn_size, params.vocab_size)
  local dropped          = nn.Dropout(params.dropout)(i[params.layers])
  local pred             = nn.LogSoftMax()(h2y(dropped))
  local err              = nn.ClassNLLCriterion()({pred, y})
  local module           = nn.gModule({x, y, prev_s},{err, nn.Identity()(next_s),pred})

  module:getParameters():uniform(-params.init_weight, params.init_weight)
  return transfer_data(module)
end

function setup()
  print("Creating a RNN LSTM network.")
  local core_network = create_network()
  paramx, paramdx = core_network:getParameters()
  model.s = {}
  model.ds = {}
  model.start_s = {}
  for j = 0, params.seq_length do
    model.s[j] = {}
    for d = 1, 2 * params.layers do
      model.s[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    end
  end
  for d = 1, 2 * params.layers do
    model.start_s[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    model.ds[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  end
  model.core_network = core_network
  model.rnns = g_cloneManyTimes(core_network, params.seq_length)
  model.norm_dw = 0
  model.err = transfer_data(torch.zeros(params.seq_length))
  model.pred=transfer_data(torch.zeros(params.vocab_size)) -- just for output, useless

end

function reset_state(state)
  if cmd_param.type == 'train' then 
    state.pos = 1
  end
  if model ~= nil and model.start_s ~= nil then
    for d = 1, 2 * params.layers do
      model.start_s[d]:zero()
    end
  end
end

function reset_ds()
  for d = 1, #model.ds do
    model.ds[d]:zero()
  end
end

function fp(state)
  g_replace_table(model.s[0], model.start_s)
  if state.pos + params.seq_length > state.data:size(1) then
    reset_state(state)  --reset string index to 1 and reset the model.s to all zero
  end
  for i = 1, params.seq_length do
    local x = state.data[state.pos]
    local y = state.data[state.pos + 1]
    local s = model.s[i - 1]
    model.err[i], model.s[i],model.pred = unpack(model.rnns[i]:forward({x, y, s}))
    state.pos = state.pos + 1
  end
  g_replace_table(model.start_s, model.s[params.seq_length])
  return model.err:mean()
end

function bp(state)
  paramdx:zero()
  reset_ds()
  for i = params.seq_length, 1, -1 do
    state.pos = state.pos - 1
    local x = state.data[state.pos]
    local y = state.data[state.pos + 1]
    local s = model.s[i - 1]
    local derr = transfer_data(torch.ones(1))
    local dummy_depred = transfer_data(torch.zeros(params.batch_size,params.vocab_size)) -- set dpred/de = zero
    local tmp = model.rnns[i]:backward({x, y, s},{derr, model.ds,dummy_depred })[3]
    g_replace_table(model.ds, tmp)
    cutorch.synchronize()
  end
  state.pos = state.pos + params.seq_length
  model.norm_dw = paramdx:norm()
  if model.norm_dw > params.max_grad_norm then
    local shrink_factor = params.max_grad_norm / model.norm_dw
    paramdx:mul(shrink_factor)
  end
  paramx:add(paramdx:mul(-params.lr))
end

function run_valid()
  reset_state(state_valid)
  g_disable_dropout(model.rnns)
  local len = (state_valid.data:size(1) - 1) / (params.seq_length)
  local perp = 0
  for i = 1, len do
    perp = perp + fp(state_valid)
  end
  print("Validation set perplexity : " .. g_f3(torch.exp(perp / len)))
  A4.ValidPerplex = g_f3(torch.exp(perp / len));
  g_enable_dropout(model.rnns)
end

function run_test()
  reset_state(state_test)
  g_disable_dropout(model.rnns)
  local perp = 0
  local len = state_test.data:size(1)
  g_replace_table(model.s[0], model.start_s)
  for i = 1, (len - 1) do
    local x = state_test.data[i]
    local y = state_test.data[i + 1]
    local s = model.s[i - 1]
    perp_tmp, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
    perp = perp + perp_tmp[1] -- what is the first element of perp_temp[1] ? 
    g_replace_table(model.s[0], model.s[1])
  end
  print("Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
  A4.TestPerplex = g_f3(torch.exp(perp / (len - 1)))
  g_enable_dropout(model.rnns)
end

function readline()
  local line = io.read("*line")
  if line == nil then error({code="EOF"}) end
  line = stringx.split(line)
  if tonumber(line[1]) == nil then 
    error({code="init"}) 
  end
  --print(#line);
  for i = 2,#line do
    if not ptb.vocab_map[line[i]] then 
      error({code="vocab", word = line[i]}) 
    end
  end
  return line
end

function query_sentences()
  while true do
    io.write("Query: len word1 word2 etc\n")
    io.flush()
    local ok, line = pcall(readline)
    if not ok then
      if line.code == "EOF" then
        break -- end loop
      elseif line.code == "vocab" then
        --print("Word not in vocabulary, only 'foo' is in vocabulary: ", line.word)
      elseif line.code == "init" then
        --print("Start with a number")
      else
        --print(line)
        --print("Failed, try again")
      end
    else
      local len = line[1]
      reset_state(state_test)
      g_disable_dropout(model.rnns)
      g_replace_table(model.s[0],model.start_s)
      local input_x = transfer_data(torch.zeros(params.batch_size))
      local y = transfer_data(torch.ones(params.batch_size))

      for i=2, #line do
        predictor = line[i]
        local idx = ptb.vocab_map[predictor]
        for i=1,params.batch_size do 
          input_x[i] = idx 
        end

        perp_tmp, model.s[1], pred_tmp = unpack(model.rnns[1]:forward({input_x, y, model.s[0]}))
        g_replace_table(model.s[0], model.s[1])
        io.write(' ',line[i])
        io.flush()
      end
      for i = 1,len do

        idx = ptb.vocab_map[predictor]
        for i=1,params.batch_size do 
          input_x[i] = idx 
        end
        local s = model.s[i - 1]
        perp_tmp, model.s[1], pred_tmp = unpack(model.rnns[1]:forward({input_x, y, model.s[0]}))
        if cmd_param.level == 'character' then
          for z=1,50 do
            io.write( pred_tmp[1][i] )
            if z<50 then
              io.write(' ')
            end
          end
          io.flush("\n")
        end
        xx = pred_tmp[1]:clone():float()
        xx = torch.multinomial(torch.exp(xx),1)
        --io.write(' ',ptb.inv_vocab_map[xx[1]])
        --io.flush()
        g_replace_table(model.s[0], model.s[1])
        predictor = ptb.inv_vocab_map[xx[1]]
      end
      --io.write('\n')
      --io.flush()
    end
  end
end
function query_sentences2()
  reset_state(state_test)
  g_disable_dropout(model.rnns)
  g_replace_table(model.s[0],model.start_s)

  while true do
    --io.write("Query: len word1 word2 etc\n")
    local STR = io.read("*line")
    local words = stringx.split(STR)

    local input_x = transfer_data(torch.zeros(params.batch_size))
    local y = transfer_data(torch.ones(params.batch_size))

    if next(words) == nil then 
      next_word='_' 
    else 
      next_word=words[1] 
    end

    input_x:fill(ptb.vocab_map[next_word])
    perp, model.s[1], pred_tmp = unpack(model.rnns[1]:forward({input_x, y, model.s[0]}))
    guess = pred_tmp[1]:clone():float()
    for i=1,guess:size(1) do
        io.write(guess[i])
        if i < guess:size(1) then
          io.write(' ')
        end
        io.flush()
    end
    -- start next line
    io.write('\n')
    io.flush()
    end
end
--function main()
print(arg)
--arg = {}
--if not arg[1] then
arg[1] = 1
--end
g_init_gpu(arg)
--state_train = {data=transfer_data(ptb.traindataset(params.batch_size))}
--state_valid =  {data=transfer_data(ptb.validdataset(params.batch_size))}
--state_test =  {data=transfer_data(ptb.testdataset(params.batch_size))}
--print("Network parameters:")
--print(params)
--io.write(cmd_param.type)
--io.flush()
if cmd_param.type == 'query' then
  io.write("OK GO\n")
  io.flush()
  query_sentences2()
else
  local states = {state_train, state_valid, state_test}
  for _, state in pairs(states) do
   reset_state(state)
  end
  setup()
  step = 0
  epoch = 0
  total_cases = 0
  beginning_time = torch.tic()
  start_time = torch.tic()
  A4={}
  print("Starting training.")
  words_per_step = params.seq_length * params.batch_size
  epoch_size = torch.floor(state_train.data:size(1) / params.seq_length)
  --perps
  while epoch < params.max_max_epoch do
   perp = fp(state_train)
   if perps == nil then
     perps = torch.zeros(epoch_size):add(perp)
   end
   perps[step % epoch_size + 1] = perp
   step = step + 1
   bp(state_train)
   total_cases = total_cases + params.seq_length * params.batch_size
   epoch = step / epoch_size
   if step % torch.round(epoch_size / 10) == 10 then
     wps = torch.floor(total_cases / torch.toc(start_time))
     since_beginning = g_d(torch.toc(beginning_time) / 60)
     print('epoch = ' .. g_f3(epoch) ..
           ', train perp. = ' .. g_f3(torch.exp(err_factor* perps:mean())) ..
           ', wps = ' .. wps ..
           ', dw:norm() = ' .. g_f3(model.norm_dw) ..
           ', lr = ' ..  g_f3(params.lr) ..
           ', since beginning = ' .. since_beginning .. ' mins.')

            A4.model = model
            A4.params = params
            if cmd_param.level == 'word' then
              torch.save("model_A4_temp.net",A4)
            else
              torch.save("modelC_A4_temp.net",A4)
            end
   end
   if step % epoch_size == 0 then
     run_valid()
     if epoch > params.max_epoch then
         params.lr = params.lr / params.decay
     end
   end
   if step % 33 == 0 then
     cutorch.synchronize()
     collectgarbage()
   end
  end

  A4.model = model
  A4.params = params
  if cmd_param.level == 'word' then
    torch.save("model_A4.net",A4)
    run_test()
  else
    torch.save("modelC_A4.net",A4)
  end
  print("Training is over.")
  query_sentences()
end
io.write('fin')
io.flush()
