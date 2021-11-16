# configfile: "config.yaml"
import numpy as np
import os
import pandas as pd
import ast

# envvars:
#     "SHELL"
consts = {"lambda_0":10e3,
                "M":1,
                "sigma_g":1,
                "t_0":0,
                "t":0.05,
                "x_0_tau":0,
                "y_0_tau":0,
                "wavelength":488,
                "n_a":1.1,
                # m:1,
                # n:1,
                "r":  np.logspace(0,2,200)
                }

consts_para = pd.DataFrame(consts);consts_para

consts_para.to_dict()
consts_para = consts_para.to_dict('records')

results = "{base_dir}/out/{consts}.csv"

all_results = expand(results,
            base_dir = workflow.basedir,
            consts=consts_para
            )

base_dir = workflow.current_basedir
script = os.path.join(workflow.basedir,"crlb_par.py")

rule all:
    input:
        all_results
        # "out/{psf_type}_{psf_width}_{signal_strength}.csv"

rule simulate:
    input:
       {script}
    # conda:
    #     "environment.yml"
    params:
        args = lambda wildcards: list(ast.literal_eval(wildcards.consts).values())
        # ok = "temp.txt"
    resources:
        mem_mb=1200
    output:
        results
    shell:
        """
	    python {input} --params {params.args} --output "{output}"
        """



# rule simulate:
#     # input:
#     #     "{basedir}/040520_pres_cluster_coins.py"
#     conda:
#         "environment.yml"
#     params:
#         # shell=os.environ["SHELL"]
#         # psf_scale="{psf_scale}"
#         # psf_type = "{psf_type}",
#         # psf_scale = "{psf_scale}",
#         # signal_strength = "{signal_strength}",
#         # thinning_type = "{thinning_type}"
#     resources:
#         mem_mb=12000

#     output:
#         directory(results)
#     shell:
#         """
# 	    python {script} \
#         --out_dir {output} \
#         --psf_type {wildcards.psf_type} \
#         --psf_scale {wildcards.psf_scale} \
#         --signal_strength {wildcards.signal_strength} \
#         --thinning_type {wildcards.thinning_type} \
#         --coin_flip_bias {wildcards.coin_flip_bias} \
#         --max_iter {wildcards.max_iter}
#         """