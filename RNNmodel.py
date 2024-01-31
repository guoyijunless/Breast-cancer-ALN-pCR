import torch
import torch.nn as nn

class ParamActivation(nn.Module):
    def __init__(self, activation_func, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.activation = activation_func
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, x):
        return self.a * self.activation(x) + self.b


class CurveFitting(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CurveFitting, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.drop = nn.Dropout()
        self.activattion = ParamActivation(nn.Sigmoid())
        self.fc2 = nn.Linear(hidden_size, output_size)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x):
        out = self.activattion(self.fc1(x))
        out = self.drop(out)
        out = self.fc2(out)
        return out

class RNNModel(nn.Module):
    def __init__(self, input_size, curve_len, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.fitline1 = CurveFitting(6, int(curve_len*2), curve_len)
        self.dropout0 = nn.Dropout()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers, dropout=0.5)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc_st = nn.Linear(hidden_size + 4, output_size)
        self.dropout1 = nn.Dropout()
        self.fc = nn.Linear(hidden_size, output_size)

        # Parameter initialization
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

        nn.init.xavier_normal_(self.attention.weight)
        nn.init.constant_(self.attention.bias, 0.0)

    def forward(self, x, st=None):
        x = torch.transpose(x, 1, 2)
        linef = self.fitline1(x)
        linef = self.dropout0(linef)
        linef = torch.transpose(linef, 1, 2)
        linef = torch.transpose(linef, 0, 1)
        out, _= self.rnn(linef)
        att = self.attention(out)
        out = out * torch.softmax(att, dim=0)
        out = torch.sum(out, dim=0)
        if st:
            out = torch.cat([out, torch.softmax(st.float(), dim=1)], dim=1)
            out = self.fc_st(out)
            return out
        out = self.dropout1(out)
        out = self.fc(out)
        return out
if __name__ == '__main__':
    net = RNNModel(10, 72, 100, 6, 2)
    x = torch.rand(1, 6, 10)
    st = torch.tensor([[0,0,1,0]])
    print(net(x).shape)