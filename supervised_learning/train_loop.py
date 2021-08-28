def train_loop(dataloader, model, loss_fn, optimizer, writer, epoch,device):
  size = len(dataloader.dataset)
  model.train()
  #print(model)
  for batch, sample in enumerate(dataloader):
    # Compute prediction and loss
    # print(batch,(X,y))
    X = sample['data'].float().to(device)
  # print(len(X))
    # print(sample['label'])
    y = sample['label'].float().to(device)
    #print(y)
    pred = model(X)

    #print('pred',pred)
    #print(y)
    loss = loss_fn(pred, y).float()
    # print(loss)
    # Backpropagation
    optimizer.zero_grad()

    loss.backward()
    # print('loss',loss)
    # nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
    '''
    for name, param in model.named_parameters():
      print('层:', name, param.size())
      print('权值梯度', torch.norm(param.grad))
      # print('权值',param)
    '''
    optimizer.step()
    # print(batch)

    if batch % 100 == 0:
      loss, current = loss.item(), batch * len(X)
      # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
  writer.add_scalar('train_loss', loss, global_step=epoch)
  # writer.add_scalar('test_loss',test_loss, global_step=epoch)