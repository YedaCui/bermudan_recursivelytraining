from deep_kolmogorov import bermudan

config = {
    "T":1,
    "num_ex":4
}
train = bermudan.Bermudan(config)

mode = "avg_bs_r"
train.recursive_training(mode, base_dir="/home/ycui/Documents/bermudan_recursivetrainning/exp_numexer_4")