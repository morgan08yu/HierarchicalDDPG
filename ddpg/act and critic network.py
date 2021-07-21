class DeterministicActorNet(nn.Module, BasicNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 action_gate,
                 action_scale,
                 gpu=False,
                 batch_norm=False,
                 non_linear=F.relu):
        super(DeterministicActorNet, self).__init__()
        self.state_dim= state_dim
        stride_time = state_dim[1] - 1 - 2  #
        features = task.state_dim[0]
        h0 = 3
        h2 = 4
        h1 = 16
        self.conv0 = nn.Conv2d(features, h0, (3, 3), stride = (1,1), padding=(1,1)) # input 64*5 *50 *10 out 64* 48 *8
        self.conv1 = nn.Conv2d(h0, h2, (3, 1)) # input 64 * 50 * 10   output 64 *48 *8
        self.conv2 = nn.Conv2d(h2, h1, (stride_time, 1), stride=(stride_time, 1))
        self.conv3 = nn.Conv2d(h1, 1, (1, 1))

        self.action_scale = action_scale
        self.action_gate = action_gate
        self.non_linear = non_linear

        if batch_norm:
            self.bn1 = nn.BatchNorm2d(h0)
            self.bn2 = nn.BatchNorm2d(h1+1)

        self.batch_norm = batch_norm
        BasicNet.__init__(self, None, gpu, False)

    def forward(self, x):
        x = self.to_torch_variable(x)
        w0 = x[:, :1, :1, :]  # weights from last step
        x = x[:, :, 1:, :]
        cash_obv = self.to_torch_variable(torch.ones((x.shape[0], x.shape[1], x.shape[2], 1)))
        x = torch.cat([cash_obv, x], -1)
        phi0 = self.non_linear(self.conv0(x))
        if self.batch_norm:
            phi0 = self.bn1(phi0)

        phi1 = self.non_linear(self.conv1(phi0))
        phi2 = self.non_linear(self.conv2(phi1))
        #h = torch.cat([phi2, w0], 1)
        action = self.conv3(phi2) # does not include cash account, add cash in next step.
        # add cash_bias before we softmax
        #cash_bias_int = 0 #
        #cash_bias = self.to_torch_variable(torch.ones(action.size())[:, :, :, :1] * cash_bias_int)
        #action = torch.cat([cash_bias, action], -1)
        batch_size = action.size()[0]
        action = action.view((batch_size, -1))
        if self.action_gate:
            action = self.action_scale * self.action_gate(action)
        #action = F.softmax(action, dim = 1)
        return action
    def predict(self, x, to_numpy=True):
        y = self.forward(x)
        if to_numpy:
            y = y.cpu().data.numpy()
        return y


class DeterministicCriticNet(nn.Module, BasicNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 gpu=False,
                 batch_norm=False,
                 non_linear=F.relu):
        super(DeterministicCriticNet, self).__init__()
        stride_time = state_dim[1] - 1 - 2  #
        self.features = features = task.state_dim[0]
        h0 = 8
        h2 = 3
        h1 = 16
        self.action = actions = action_dim
        self.conv0 = nn.Conv2d(features, h0, (3, 3), padding=(1,1))
        self.conv1 = nn.Conv2d(h0, h2, (3, 1))
        self.conv2 = nn.Conv2d(h2, h1, (stride_time, 1), stride=(stride_time, 1))
        self.layer3 = nn.Linear((h1+1) * actions, 1)
        self.non_linear = non_linear

        if batch_norm:
            self.bn1 = nn.BatchNorm2d(h0)
            self.bn2 = nn.BatchNorm2d(h1+1)
        self.batch_norm = batch_norm

        BasicNet.__init__(self, None, gpu, False)

    def forward(self, x, action):
        x = self.to_torch_variable(x)
        actions = self.to_torch_variable(action)[:, None, None, :]  # remove cash bias

        w0 = x[:, :1, :1, :]  # weights from last step
        x = x[:, :, 1:, :]
        cash_obv = self.to_torch_variable(torch.ones((x.shape[0], x.shape[1], x.shape[2], 1)))
        x = torch.cat([cash_obv, x], -1)
        phi0 = self.non_linear(self.conv0(x))
        if self.batch_norm:
            phi0 = self.bn1(phi0)
        phi1 = self.non_linear(self.conv1(phi0))
        phi2 = self.non_linear(self.conv2(phi1))
        h = torch.cat([phi2, actions], 1)
        if self.batch_norm:
            h = self.bn2(h)
        batch_size = x.size()[0]
        #h = self.non_linear(self.layer3(phi2))
        sc = self.layer3(h.view((batch_size, -1)))
        return sc

    def predict(self, x, action):
        return self.forward(x, action)



    class DeterministicActorNet(nn.Module, BasicNet):
        def __init__(self,
                     state_dim,
                     action_dim,
                     action_gate,
                     action_scale,
                     gpu=False,
                     batch_norm=False,
                     non_linear=F.relu):
            super(DeterministicActorNet, self).__init__()
            self.state_dim = state_dim
            stride_time = state_dim[1] - 1 - 2  #
            features = task.state_dim[0]
            h0 = 3
            h2 = 4
            h1 = 16
            self.conv0 = nn.Conv2d(features, h0, (3, 3), stride=(1, 1),
                                   padding=(1, 1))  # input 64*5 *50 *10 out 64* 48 *8
            self.conv1 = nn.Conv2d(h0, h2, (3, 1))  # input 64 * 50 * 10   output 64 *48 *8
            self.conv2 = nn.Conv2d(h2, h1, (stride_time, 1), stride=(stride_time, 1))
            self.conv3 = nn.Conv2d(h1, 1, (1, 1))

            self.action_scale = action_scale
            self.action_gate = action_gate
            self.non_linear = non_linear

            if batch_norm:
                self.bn1 = nn.BatchNorm2d(h0)
                self.bn2 = nn.BatchNorm2d(h1 + 1)

            self.batch_norm = batch_norm
            BasicNet.__init__(self, None, gpu, False)

        def forward(self, x):
            x = self.to_torch_variable(x)
            w0 = x[:, :1, :1, :]  # weights from last step
            x = x[:, :, 1:, :]
            cash_obv = self.to_torch_variable(torch.ones((x.shape[0], x.shape[1], x.shape[2], 1)))
            x = torch.cat([cash_obv, x], -1)
            phi0 = self.non_linear(self.conv0(x))
            if self.batch_norm:
                phi0 = self.bn1(phi0)

            phi1 = self.non_linear(self.conv1(phi0))
            phi2 = self.non_linear(self.conv2(phi1))
            # h = torch.cat([phi2, w0], 1)
            action = self.conv3(phi2)  # does not include cash account, add cash in next step.
            # add cash_bias before we softmax
            # cash_bias_int = 0 #
            # cash_bias = self.to_torch_variable(torch.ones(action.size())[:, :, :, :1] * cash_bias_int)
            # action = torch.cat([cash_bias, action], -1)
            batch_size = action.size()[0]
            action = action.view((batch_size, -1))
            if self.action_gate:
                action = self.action_scale * self.action_gate(action)
            # action = F.softmax(action, dim = 1)
            return action

        def predict(self, x, to_numpy=True):
            y = self.forward(x)
            if to_numpy:
                y = y.cpu().data.numpy()
            return y
    class DeterministicCriticNet(nn.Module, BasicNet):
        def __init__(self,
                     state_dim,
                     action_dim,
                     gpu=False,
                     batch_norm=False,
                     non_linear=F.relu):
            super(DeterministicCriticNet, self).__init__()
            stride_time = state_dim[1] - 1 - 2  #
            self.features = features = task.state_dim[0]
            h0 = 8
            h2 = 3
            h1 = 16
            self.action = actions = action_dim
            self.conv0 = nn.Conv2d(features, h0, (3, 3), padding=(1, 1))
            self.conv1 = nn.Conv2d(h0, h2, (3, 1))
            self.conv2 = nn.Conv2d(h2, h1, (stride_time, 1), stride=(stride_time, 1))
            self.layer3 = nn.Linear((h1 + 1) * actions, 1)
            self.non_linear = non_linear

            if batch_norm:
                self.bn1 = nn.BatchNorm2d(h0)
                self.bn2 = nn.BatchNorm2d(h1 + 1)
            self.batch_norm = batch_norm

            BasicNet.__init__(self, None, gpu, False)

        def forward(self, x, action):
            x = self.to_torch_variable(x)
            actions = self.to_torch_variable(action)[:, None, None, :]  # remove cash bias

            w0 = x[:, :1, :1, :]  # weights from last step
            x = x[:, :, 1:, :]
            cash_obv = self.to_torch_variable(torch.ones((x.shape[0], x.shape[1], x.shape[2], 1)))
            x = torch.cat([cash_obv, x], -1)
            phi0 = self.non_linear(self.conv0(x))
            if self.batch_norm:
                phi0 = self.bn1(phi0)
            phi1 = self.non_linear(self.conv1(phi0))
            phi2 = self.non_linear(self.conv2(phi1))
            h = torch.cat([phi2, actions], 1)
            if self.batch_norm:
                h = self.bn2(h)
            batch_size = x.size()[0]
            # h = self.non_linear(self.layer3(phi2))
            sc = self.layer3(h.view((batch_size, -1)))
            return sc

        def predict(self, x, action):
            return self.forward(x, action)


    class DeterministicActorNet(nn.Module, BasicNet):
        def __init__(self,
                     state_dim,
                     action_dim,
                     action_gate,
                     action_scale,
                     gpu=False,
                     batch_norm=False,
                     non_linear=F.relu):
            super(DeterministicActorNet, self).__init__()

            stride_time = state_dim[1] - 1 - 2  #
            features = task.state_dim[0]
            h0 = 4
            h2 = 3
            h1 = 8
            self.conv0 = nn.Conv2d(features, h0, (3, 3), stride=(1, 1),
                                   padding=(1, 1))  # input 64*5 *50 *10 out 64* 48 *8
            self.conv1 = nn.Conv2d(h0, h2, (3, 1))  # input 64 * 50 * 10   output 64 *48 *8
            self.conv2 = nn.Conv2d(h2, h1, (stride_time, 1), stride=(stride_time, 1))
            self.conv3 = nn.Conv2d((h1 + 1), 1, (1, 1))

            self.action_scale = action_scale
            self.action_gate = action_gate
            self.non_linear = non_linear

            if batch_norm:
                self.bn1 = nn.BatchNorm2d(h0)
                self.bn2 = nn.BatchNorm2d(h1 + 1)

            self.batch_norm = batch_norm
            BasicNet.__init__(self, None, gpu, False)

        def forward(self, x):
            x = self.to_torch_variable(x)
            w0 = x[:, :1, :1, :]  # weights from last step
            x = x[:, :, 1:, :]

            phi0 = self.non_linear(self.conv0(x))
            if self.batch_norm:
                phi0 = self.bn1(phi0)

            phi1 = self.non_linear(self.conv1(phi0))
            phi2 = self.non_linear(self.conv2(phi1))
            h = torch.cat([phi2, w0], 1)
            if self.batch_norm:
                h = self.bn2(h)
            phi3 = self.conv3(h)  # does not include cash account, add cash in next step.
            # add cash_bias before we softmax
            cash_bias_int = 0  #
            cash_bias = self.to_torch_variable(torch.ones(phi3.size())[:, :, :, :1] * cash_bias_int)
            action = torch.cat([cash_bias, phi3], -1)
            batch_size = action.size()[0]
            action = action.view((batch_size, -1))
            if self.action_gate:
                action = self.action_scale * self.action_gate(action)
            action = F.softmax(action, dim=1)
            return action

        def predict(self, x, to_numpy=True):
            y = self.forward(x)
            if to_numpy:
                y = y.cpu().data.numpy()
            return y
    class DeterministicCriticNet(nn.Module, BasicNet):
        def __init__(self,
                     state_dim,
                     action_dim,
                     gpu=False,
                     batch_norm=False,
                     non_linear=F.relu):
            super(DeterministicCriticNet, self).__init__()
            stride_time = state_dim[1] - 1 - 2  #
            self.features = features = task.state_dim[0]
            h0 = 4
            h2 = 4
            h1 = 8
            self.action = actions = action_dim - 1
            self.conv0 = nn.Conv2d(features, h0, (3, 3), padding=(1, 1))
            self.conv1 = nn.Conv2d(h0, h2, (3, 1))
            self.conv2 = nn.Conv2d(h2, h1, (stride_time, 1), stride=(stride_time, 1))
            self.layer3 = nn.Linear((h1 + 2) * actions, 1)
            self.non_linear = non_linear

            if batch_norm:
                self.bn1 = nn.BatchNorm2d(h0)
                self.bn2 = nn.BatchNorm2d(h1 + 2)
            self.batch_norm = batch_norm

            BasicNet.__init__(self, None, gpu, False)

        def forward(self, x, action):
            x = self.to_torch_variable(x)
            action = self.to_torch_variable(action)[:, None, None, :-1]  # remove cash bias

            w0 = x[:, :1, :1, :]  # weights from last step
            x = x[:, :, 1:, :]

            phi0 = self.non_linear(self.conv0(x))
            if self.batch_norm:
                phi0 = self.bn1(phi0)
            phi1 = self.non_linear(self.conv1(phi0))
            phi2 = self.non_linear(self.conv2(phi1))
            h = torch.cat([phi2, w0, action], 1)
            if self.batch_norm:
                h = self.bn2(h)
            batch_size = x.size()[0]
            # action = self.non_linear(self.layer3(h))
            action = self.layer3(h.view((batch_size, -1)))
            return action

        def predict(self, x, action):
            return self.forward(x, action)

    def to_torch_variable(x, dtype='float32'):
        if isinstance(x, Variable):
            return x
        if not isinstance(x, torch.FloatTensor):
           x = torch.from_numpy(np.asarray(x, dtype=dtype))
        return Variable(x)