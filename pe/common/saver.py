import os
from datetime import datetime
import torch
from pe.common import util


class Saver:

  def __init__(self, dir, keep_all=False):
    self.dir = dir
    self.best_acc = 0
    self.best_file = ''
    bests = self.get_best_acc()
    if bests:
      self.best_acc = bests[1]
      self.best_file = bests[0]
    self.last_file_name = self.get_last_file_name()
    self.keep_all = keep_all

  def get_last_file_name(self):
    for file in os.listdir(self.dir):
      if (not file.startswith('best')) and file.endswith('.pkl'):
        return file

  def get_best_acc(self):
    for file in os.listdir(self.dir):
      if file.startswith('best'):
        return file, float(file[4:-4])

  def store(self, nsample, model, optimizer, validate_acc, extra=''):
    content = {
      'n_sample_trained': nsample,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'validate_acc': validate_acc,
      'git': util.get_git_revision_hash(),
      'extra': extra
      }
    file_name = '_'.join(
        (datetime.now().strftime("%Y-%m-%d-%H%M"),
        str(round(validate_acc, 2)) + '_' +\
          util.get_git_revision_hash(True) + '.pkl')
      )
    torch.save(content, os.path.join(self.dir, file_name))
    if self.last_file_name:
      last_file_path = os.path.join(self.dir, self.last_file_name)
      if os.path.exists(last_file_path) and (not self.keep_all):
        os.remove(last_file_path)
    self.last_file_name = file_name
    if validate_acc > self.best_acc:
      best_file = f'best{round(validate_acc, 2)}.pkl'
      torch.save(content, os.path.join(self.dir, best_file))
      if self.best_file:
        best_file_path = os.path.join(self.dir, self.best_file)
        if os.path.exists(best_file_path):
          os.remove(best_file_path)
      self.best_file = best_file
      self.best_acc = validate_acc
