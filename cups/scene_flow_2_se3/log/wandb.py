def init(project_name, entity_name, dir, run_id, args):
    # e.g.
    # project_name = sflow2rigid3d
    # entity_name = limpbot
    # dir = results/wandb
    import wandb

    wandb.init(project=project_name, entity=entity_name, dir=dir, id=run_id)
    wandb.config.update(args)
    # config={}
    # wandb.config.update({"lr": 0.1, "channels": 16})


def log_metrics(metrics, step):
    # dict: key, val (val might be torch tensor)
    # wandb.log(dict, step=step) : steps must be increasing 0, 1, 2...
    # wandb.log(dict) : steps automatically increase
    import wandb

    wandb.log(metrics, step=step)


def log_table(table, table_key, step):
    # list[lists[], list[]]: key, val (val might be torch tensor)
    # headers: table[0]
    # columns = table[1:]
    import wandb

    wandb_table = wandb.Table(data=table[1:], columns=table[0])
    wandb.log({table_key: wandb_table}, step)


def delete_all_runs(entity_name, project_name):
    import wandb

    api = wandb.Api()
    runs = api.runs(entity_name + "/" + project_name)
    for run in runs:
        run_name = run.id
        run_single = api.run(entity_name + "/" + project_name + "/" + run_name)
        run_single.delete()
    # run = api.run("limpbot/<project>/<run_id>")
    # run.delete()
