import os
lr_rates = [.1 .5, .01, .001, .005]
hidden_sizes = [100, 200, 500]
dropouts = [.5,.75,.8]

for lr_rate in lr_rates:
    for hidden_size in hidden_sizes:
        for dropout in dropouts:
            str = "python3 setup.py  --lr_rate {0}  --hid_archi {1} --dropout {2}".format(lr_rate,hidden_size,dropout)
            os.system(str)
