import os
from shutil import copy

import torch

import cups.scene_flow_2_se3.log.wandb as o4log_wandb
import cups.scene_flow_2_se3.vision.visualization as o4visual


class Logger:

    def __init__(self, args):
        self.results_dir = args.setup_results_dir

        datetime_now = args.datetime
        date_string = datetime_now.strftime("%d_%m_%Y__%H_%M_%S")
        run_id = "_".join(args.data_dataset_tags) + "_" + args.sflow2se3_approach + "__" + date_string
        self.run_dir = os.path.join(args.setup_results_dir, run_id)
        os.mkdir(self.run_dir)
        os.mkdir(os.path.join(self.run_dir, "imgs"))
        os.mkdir(os.path.join(self.run_dir, "configs"))

        dict_args = vars(args)
        for key, val in dict_args.items():
            if key.startswith("config") and val is not None:
                dst_dir = os.path.join(self.run_dir, os.path.dirname(val))
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                copy(os.path.join(os.getcwd(), val), os.path.join(self.run_dir, val))
        self.wandb_log = args.setup_wandb_log
        self.wandb_dir = os.path.join(args.setup_results_dir, "wandb")
        if not self.wandb_log:
            os.environ["WANDB_MODE"] = "offline"

        o4log_wandb.init(
            project_name=args.setup_wandb_project,
            entity_name=args.setup_wandb_entity,
            dir=self.wandb_dir,
            run_id=run_id,
            args=args,
        )

    def log_image(self, img, key, step=None):
        fpath = os.path.join(self.run_dir, "imgs", key + ".png")
        fdir = os.path.dirname(fpath)
        fname = os.path.basename(fpath)

        if not os.path.exists(fdir):
            os.makedirs(fdir)

        if not key.split("_")[-1].isdigit():

            fdir = os.path.dirname(fpath)
            fdir_fnames_digits = [
                int(fdir_fname.split(".")[-2].split("_")[-1])
                for fdir_fname in os.listdir(fdir)
                if fdir_fname.startswith(fname.split(".")[-2])
            ]
            if len(fdir_fnames_digits) == 0:
                digit = 0
            else:
                digit = max(fdir_fnames_digits) + 1
            fpath = os.path.join(self.run_dir, "imgs", key + "_" + str(digit) + ".png")

        o4visual.visualize_img(img, fpath=fpath)

    def metrics_2_avg(self, metrics):
        metrics_avg = {}
        for key, val in metrics.items():
            if isinstance(val, list):
                val = torch.stack(val).mean()
            if isinstance(val, torch.Tensor):
                val = val.item()
            metrics_avg[key] = val
        return metrics_avg

    def log_metrics(self, metrics, step):

        metrics_avg = self.metrics_2_avg(metrics)

        if self.wandb_log:
            o4log_wandb.log_metrics(metrics_avg, step)

    def log_table(self, table, table_key=None, step=None, remove_old=False):

        self.print_table(table)

        if table_key is not None:
            self.save_table(table, table_key, remove_old=remove_old)
            if self.wandb_log:
                o4log_wandb.log_table(table, table_key, step)

    def print_table(self, table):
        from tabulate import tabulate

        print(tabulate(table[1:], headers=table[0]))

    def save_table(self, table, table_key, remove_old=False):
        fpath = os.path.join(self.results_dir, table_key + ".csv")
        import csv

        if os.path.isfile(fpath) and remove_old is False:
            open_method = "a"
            table = table[1:]
        else:
            open_method = "w"
        with open(fpath, open_method, newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            for row in table:
                csv_writer.writerow(row)
