import os
def read_file(file_slug):

  file_path = os.path.join("repositories", file_slug)
  print(file_path)
  with open(file_path, "r") as file:
    return file.read()