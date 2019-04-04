import tensorflow as tf
import os

def save_checkpoint(sess, is_best, opts, logger, epoch, criterion="dev"):
    """
    Save current model to checkpoints dir
    """
    # --- borrowed from: https://github.com/pytorch/examples/blob/master/imagenet/main.py#L139
    if is_best:
        out_file = os.path.join(
            opts.checkpoints, "".join(["best_under", criterion, "criterion.ckpt"])
        )
        # torch.save(state, out_file)
        save(sess, out_file)
        logger.info("Best model saved to {} at epoch {}".format(out_file, str(epoch)))
    else:
        out_file = os.path.join(opts.checkpoints, "checkpoint.ckpt")
        # torch.save(state, out_file)
        save(sess, out_file)
        logger.info("Checkpoint saved to {} at epoch {}".format(out_file, str(epoch)))
    return out_file

def save(sess, out_file):
    """
    Saves the current session to a checkpoint
    """

    saver = tf.train.Saver(max_to_keep=1)
    save_path = saver.save(sess, out_file)
    return save_path

def restore(sess, out_file, var_dict=None):
    """
    Restores a session from a checkpoint
    :param sess: current session instance
    :param model_path: path to file system checkpoint location
    """
    try:
        if var_dict is None:
            saver = tf.train.Saver()
        else:
            saver = tf.train.Saver(var_list=var_dict)
        # saver.restore(sess, model_path+"checkpoints/model.ckpt")
        saver.restore(sess, out_file)
        print("Model restored from file: %s" % out_file)
        return True
    except Exception as e:
        print("Cant restore from path {}".format(out_file))
        print(e)
        return False


def restore_from_best(sess, model_path, var_dict=None):
    """
    Restores a session from a checkpoint
    :param sess: current session instance
    :param model_path: path to file system checkpoint location
    """
    try:
        if var_dict is None:
            saver = tf.train.Saver()
        else:
            saver = tf.train.Saver(var_list=var_dict)
        saver.restore(sess, model_path+"/checkpoints/best/model.ckpt")
        print("Model restored from file: %s" % model_path)
        return True
    except Exception as e:
        print("Cant restore the best from path {}".format(model_path))
        return False