from __future__ import print_function
from __future__ import division

import numpy as np
from collections import OrderedDict
import argparse
import os

# from math import log
import multiprocessing
import logging


class Arguments(object):
    """
    Inspired on https://github.com/lquirosd/P2PaLA/blob/master/utils/optparse.py
    """

    def __init__(self, logger=None):
        """
        """
        self.logger = logger or logging.getLogger(__name__)
        parser_description = """
        
        """

        n_cpus = multiprocessing.cpu_count()

        self.parser = argparse.ArgumentParser(
            description=parser_description,
            fromfile_prefix_chars="@",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self.parser.convert_arg_line_to_args = self._convert_file_to_args
        # ----------------------------------------------------------------------
        # ----- Define general parameters
        # ----------------------------------------------------------------------
        general = self.parser.add_argument_group("General Parameters")

        # general.add_argument(
        #     "--config", default=None, type=str, help="Use this configuration file"
        # )
        general.add_argument(
            "--exp_name",
            default="author",
            type=str,
            help="""Name of the experiment. Models and data 
                                               will be stored into a folder under this name""",
        )

        general.add_argument(
            "--work_dir", default="./work/", type=str, help="Where to place output data"
        )

        general.add_argument(
            "--represent", default="tfidf", type=str, help="Representation of the data. tfidf or WE (Word embeddings)"
        )

        # ----------------------------------------------------------------------
        # ----- Define processing data parameters
        # ----------------------------------------------------------------------
        data = self.parser.add_argument_group("Data Related Parameters")
        general.add_argument(
            "--model", default="FF", type=str, help="FF (MLP), CNN or RNN network model"
        )

        general.add_argument(
            "--lang", default="es", type=str, help="language es/en"
        )

        data.add_argument(
            "--filters",
            default=[32,64,128],
            type=type([]),
            help="Filters for the CNN network",
        )
        data.add_argument(
            "--layers",
            default=[32,64,128],
            type=type([]),
            help="Layers for the FF or RNN network",
        )
        data.add_argument(
            "--min_ngram", default=1, type=int, help="Min ngram"
        )

        data.add_argument(
            "--max_ngram", default=3, type=int, help="Max ngram"
        )

        data.add_argument(
            "--max_features", default=10000, type=int, help="Limit the vocabulary (tfidf)"
        )

        # ----------------------------------------------------------------------
        # ----- Define dataloader parameters
        # ----------------------------------------------------------------------
        loader = self.parser.add_argument_group("Data Loader Parameters")
        loader.add_argument(
            "--batch_size", default=32, type=int, help="Number of images per mini-batch"
        )

        # ----------------------------------------------------------------------
        # ----- Define Optimizer parameters
        # ----------------------------------------------------------------------
        optim = self.parser.add_argument_group("Optimizer Parameters")
        optim.add_argument(
            "--adam_lr",
            default=0.001,
            type=float,
            help="Initial Lerning rate for ADAM opt",
        )
        optim.add_argument(
            "--adam_beta1",
            default=0.5,
            type=float,
            help="First ADAM exponential decay rate",
        )
        optim.add_argument(
            "--adam_beta2",
            default=0.999,
            type=float,
            help="Secod ADAM exponential decay rate",
        )

        # ----------------------------------------------------------------------
        # ----- Define Train parameters
        # ----------------------------------------------------------------------
        train = self.parser.add_argument_group("Training Parameters")
        tr_meg = train.add_mutually_exclusive_group(required=False)
        tr_meg.add_argument(
            "--do_train",
            dest="do_train",
            help="Run train stage",
            type=self.str2bool,
            default=True
        )
        train.add_argument(
            "--dataset",
            default="PAN2019",
            type=str,
            help="PAN2019 or canon60",
        )
        tr_meg.set_defaults(do_train=True)
        train.add_argument(
            "--cont_train",
            default=False,
            type=self.str2bool,
            help="Continue training using this model",
        )
        train.add_argument(
            "--prev_model",
            default=None,
            type=str,
            help="Use this previously trained model",
        )
        train.add_argument(
            "--save_rate",
            default=10,
            type=int,
            help="Save checkpoint each --save_rate epochs",
        )
        train.add_argument(
            "--i",
            default="/data2/jose/data/pan19-author-profiling-training-2019-02-18/es",
            type=str,
            help="""Train and test data folder with truth files""",
        )
        train.add_argument(
            "--file_i",
            default="/home/prietofontcuberta19/datos/es",
            type=str,
            help="""Train and test data folder with truth files""",
        )
        train.add_argument(
            "--epochs", default=50, type=int, help="Number of training epochs"
        )

        # ----------------------------------------------------------------------
        # ----- Define Test parameters
        # ----------------------------------------------------------------------
        test = self.parser.add_argument_group("Test Parameters")
        te_meg = test.add_mutually_exclusive_group(required=False)
        te_meg.add_argument(
            "--do_test", dest="do_test", type=self.str2bool, help="Run test stage",
            default=True
        )

        te_meg.set_defaults(do_test=True)

        test.add_argument(
            "--te_data",
            default="./data/test/",
            type=str,
            help="""Test data folder. Test images are
                                         expected there, also PAGE XML files are
                                         expected to be in --te_data/page folder
                                         """,
        )
        test.add_argument(
            "--o",
            default="/data2/jose/data/pan19-author-profiling-training-2019-02-18/output_es",
            type=str,
            help="""Test output folder. It will be created if doesnt exists""",
        )

        # ----------------------------------------------------------------------
        # ----- Define Validation parameters
        # ----------------------------------------------------------------------
        validation = self.parser.add_argument_group("Validation Parameters")
        v_meg = validation.add_mutually_exclusive_group(required=False)
        v_meg.add_argument(
            "--do_val", dest="do_val", type=self.str2bool, help="Run Validation stage",
            default=True
        )
        v_meg.set_defaults(do_val=True)
        validation.add_argument(
            "--val_data",
            default="./data/val/",
            type=str,
            help="""Validation data folder. Validation images are
                                         expected there, also PAGE XML files are
                                         expected to be in --te_data/page folder
                                         """,
        )

    def str2bool(self, v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    def _convert_file_to_args(self, arg_line):
        return arg_line.split(" ")

    def __str__(self):
        """pretty print handle"""
        data = "------------ Options -------------"
        try:
            for k, v in sorted(vars(self.opts).items()):
                data = data + "\n" + "{0:15}\t{1}".format(k, v)
        except:
            data = data + "\nNo arguments parsed yet..."

        data = data + "\n---------- End  Options ----------\n"
        return data

    def __repr__(self):
        return self.__str__()

    def _check_out_dir(self, pointer):
        """ Checks if the dir is writable"""
        if os.path.isdir(pointer):
            # --- check if is writeable
            if os.access(pointer, os.W_OK):
                if not (os.path.isdir(pointer + "/checkpoints")):
                    os.makedirs(pointer + "/checkpoints")
                    self.logger.info(
                        "Creating checkpoints dir: {}".format(pointer + "/checkpoints")
                    )
                return pointer
            else:
                raise argparse.ArgumentTypeError(
                    "{} folder is not writeable.".format(pointer)
                )
        else:
            try:
                os.makedirs(pointer)
                self.logger.info("Creating output dir: {}".format(pointer))
                os.makedirs(pointer + "/checkpoints")
                self.logger.info(
                    "Creating checkpoints dir: {}".format(pointer + "/checkpoints")
                )
                return pointer
            except OSError as e:
                raise argparse.ArgumentTypeError(
                    "{} folder does not exist and cannot be created\n".format(e)
                )


    def output_dir(self, pointer):
        """ Checks if the dir is writable"""
        if os.path.isdir(pointer):
            # --- check if is writeable
            if os.access(pointer, os.W_OK):
                return pointer
            else:
                raise argparse.ArgumentTypeError(
                    "{} folder is not writeable.".format(pointer)
                )
        else:
            try:
                os.makedirs(pointer)
                self.logger.info("Creating output dir: {}".format(pointer))
                return pointer
            except OSError as e:
                raise argparse.ArgumentTypeError(
                    "{} folder does not exist and cannot be created\n".format(e)
                )
    def parse(self):
        """Perform arguments parsing"""

        self.opts, unkwn = self.parser.parse_known_args()
        self._check_out_dir(self.opts.work_dir)
        self.output_dir(self.opts.o)
        self.opts.log_file = self.opts.work_dir + "/" + self.opts.exp_name + ".log"
        self.opts.checkpoints = os.path.join(self.opts.work_dir, "checkpoints/")
        return self.opts
