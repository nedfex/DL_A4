
require 'nn'
require 'nngraph'

function lstm(x, prev_c, prev_h)
  local function new_input_sum()
    local xt            = nn.Linear(params.rnn_size, params.rnn_size)
    local ht_1          = nn.Linear(params.rnn_size, params.rnn_size)
    local c_t_1         = nn.Linear(params.rnn_size, params.rnn_size)
    return nn.CAddTable()({xt(i), ht_1(prev_h) , c_t_1(prev_c) })
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
