import subprocess


def get_git_revision_hash(short=False) -> str:
  command_list = ['git', 'rev-parse']
  if short:
    command_list.append('--short')
  command_list.append('HEAD')
  return subprocess.check_output(command_list).decode('ascii').strip()


def get_model_device(model):
  return next(model.parameters()).device
