import os, sys

def create_structure(dir_name):
    """
    Make a structure
    :param structure:
    :return:
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    if not os.path.exists(dir_name+"/checkpoints"):
        os.makedirs(dir_name+"/checkpoints")

    if not os.path.exists(dir_name+"/checkpoints/best"):
        os.makedirs(dir_name+"/checkpoints/best")

    # if not os.path.exists(dir_name+"/output"):
    #     os.makedirs(dir_name+"/output")


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)