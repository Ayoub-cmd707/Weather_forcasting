import os

#check added value of m_mulp (as suggested by reviewers)

base_command = "python3 Train.py --loss {loss} --ens-num 11 --target-var {target_var} --lr 0.001 --epochs 50 --batch-size 2 --nheads {nheads} --num_blocks {num_blocks} --projection_channels {projection_channels} --mlp_mult {mlp_mult} --num_predictors={num_predictors} --lambda_reg={lambda_reg} --k_reg={k_reg}"

configs = [

    (8, 4, 64, 4, 0.02, 2.3),
    (8, 4, 64, 4, 0.02, 2.3),
    (8, 4, 64, 4, 0.04, 2.3),
    (8, 4, 64, 4, 0.04, 3.3),

]


target_vars = {
    "ssrd6": ("CRPSKERNELSTEP", 18),
}


for target_var, (loss, num_predictors) in target_vars.items():
    for nheads, num_blocks, projection_channels, mlp_mult,lambda_reg,k_reg in configs:
        command = base_command.format(
            loss=loss,
            target_var=target_var,
            nheads=nheads,
            num_blocks=num_blocks,
            projection_channels=projection_channels,
            mlp_mult=mlp_mult,
            num_predictors=num_predictors,
            lambda_reg=lambda_reg,
            k_reg=k_reg
        )
        print(f"Executing: {command}")
        os.system(command)