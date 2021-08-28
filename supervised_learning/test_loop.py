import torch


def test_loop(dataloader, model, loss_fn, writer, epoch,device):
  size = len(dataloader.dataset)
  test_loss, correct = 0, 0
  count = 0
  with torch.no_grad():
    for sample in dataloader:
      X = sample['sound'].float().to(device)

      # print(len(X))
      y = sample['label'].float().to(device)
      pred = model(X)
      #print('y',y)
      #print('pred',pred*180)
      #print(y.size()[0])
      #plot_and_accuracy(y.size()[0],y.cpu(),pred.cpu().flatten()*180)
      test_loss += loss_fn(pred, y).item()
      count += 1

      # print(count)
      correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
      print(correct)
  test_loss /= count
  correct /= size
  print('accuracy',correct)
  # writer.add_scalar('train_loss', train_loss, global_step=epoch)
  writer.add_scalar('test_loss', test_loss, global_step=epoch)
  # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")