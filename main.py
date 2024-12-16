from deep_kolmogorov import bermudan

config = {
    "T":1,
    "num_ex":2
}
train = bermudan.Bermudan(config)

mode = "avg_bs_basket"
train.recursive_training(mode, base_dir="/home/ycui/Documents/bermudan_recursivetrainning/10D_exp_numexer_2")