require 'torch'
cmd = torch.CmdLine()
cmd:option('-type','query','Executing Type, query or train')
cmd:text()

cmd_param = cmd:parse(arg)
print(cmd_param .type)